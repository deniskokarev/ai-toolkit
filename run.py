import os
import sys
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()
os.environ["HF_XET_HIGH_PERFORMANCE"] = os.getenv("HF_XET_HIGH_PERFORMANCE", "1")
os.environ["HF_HUB_DISABLE_XET"] = os.getenv("HF_HUB_DISABLE_XET", "0")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
seed = None
if "SEED" in os.environ:
    try:
        seed = int(os.environ["SEED"])
        # under distributed data-parallel training each rank must shuffle its
        # batches differently, or the replicas compute identical gradients and
        # the extra GPUs do duplicated work
        seed += int(os.environ.get("RANK", 0))
    except ValueError:
        print(f"Invalid SEED value: {os.environ['SEED']}. SEED must be an integer.")

sys.path.insert(0, os.getcwd())

# The UI launches jobs with no console; keep anything we shell out to (torch
# compiles, HF git downloads) from flashing a console window. Must come before
# any import that might spawn a subprocess.
from toolkit.win_console import suppress_child_consoles
suppress_child_consoles()

# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'


def _maybe_relaunch_ddp():
    """Multi-GPU data-parallel self-launch.

    If a config lists ``ddp_devices`` (process level, e.g.
    ``ddp_devices: ["cuda:0", "cuda:1"]``) and we are not already inside a
    distributed launch, re-exec through ``accelerate launch`` with one
    process per listed device. Each process trains a full model replica on
    its own GPU with per-rank batches; gradients are all-reduced by
    accelerate/DDP. Runs before any torch import so the relaunch is instant.
    """
    if os.environ.get("LOCAL_RANK") is not None or os.environ.get("WORLD_SIZE") is not None:
        return  # already inside a distributed launch
    import yaml
    devices = None
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ("-n", "--name", "-l", "--log"):
            skip_next = True
            continue
        if arg.startswith("-"):
            continue
        path = arg
        if not os.path.exists(path):
            for ext in (".yaml", ".yml", ".json"):
                candidate = os.path.join("config", arg + ext)
                if os.path.exists(candidate):
                    path = candidate
                    break
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
        except Exception:
            continue
        for proc in ((cfg or {}).get("config", {}) or {}).get("process", []) or []:
            dd = proc.get("ddp_devices") if isinstance(proc, dict) else None
            if dd and len(dd) > 1:
                devices = [str(d) for d in dd]
                break
        if devices:
            break
    if not devices:
        return
    env = dict(os.environ)
    if not any(k in env for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
        # map the listed devices to the visible set; ranks then bind to
        # cuda:LOCAL_RANK within it. If the user already restricted
        # visibility, respect their mapping and just launch N processes.
        indices = [d.split(":", 1)[1] if ":" in d else d for d in devices]
        env["HIP_VISIBLE_DEVICES"] = ",".join(indices)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(indices)
    # invoke accelerate through our own interpreter: the UI launches this
    # script with an absolute python path, so venv/bin may not be on PATH
    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", str(len(devices)),
        "--num_machines", "1",
        "--mixed_precision", "no",
        "--dynamo_backend", "no",
        os.path.abspath(__file__),
    ] + sys.argv[1:]
    print(
        f"ddp_devices {devices}: relaunching via accelerate launch "
        f"--num_processes {len(devices)}",
        flush=True,
    )
    os.execvpe(cmd[0], cmd, env)


_maybe_relaunch_ddp()

# set torch to trace mode
import torch

# MIOpen picks pathological conv algorithms on RDNA3/RDNA4 (36x on a 1536px
# VAE encode) -- must happen before any model is built. See toolkit/rocm.py.
from toolkit.rocm import configure_rocm_backends
configure_rocm_backends()

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    torch.autograd.set_detect_anomaly(True)

if seed is not None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )
    args = parser.parse_args()
    
    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_acc("")
                print_acc("========================================")
                print_acc("Job stopped")
                print_acc("========================================")
                sys.exit(0)


if __name__ == '__main__':
    main()
