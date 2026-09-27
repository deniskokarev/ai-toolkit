"""Import a training config (YAML or JSON) as a UI job.

    python scripts/import_job.py [-u] <name> <config_yaml>

Creates job <name> from the config, the same way the UI's "Import Config"
button would, and leaves it stopped for you to start or queue. If a job with
that name exists, -u replaces its config in place (typically to raise steps /
lr / batch size before resuming); the job keeps its id, queue position and
GPUs, and a diff of the changed keys is printed. A job that is running,
queued or stopping is never updated.

The job's name becomes config.name, so it trains into
<TRAINING_FOLDER>/<name> and resumes from any checkpoints already there.
"""
import argparse
import json
import os
import re
import sys

import yaml

import aitk_ui

ACTIVE = ("running", "queued", "stopping")


def load_config(path):
    with open(path) as f:
        text = f.read()
    if path.endswith((".json", ".jsonc")):
        return json.loads(text)
    return yaml.safe_load(text)


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out or {prefix: {}}
    if isinstance(obj, list) and any(isinstance(v, (dict, list)) for v in obj):
        out = {}
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
        return out or {prefix: []}
    return {prefix: obj}


def print_diff(old, new):
    a, b = flatten(old), flatten(new)
    changed = [k for k in sorted(a.keys() | b.keys()) if a.get(k, "<unset>") != b.get(k, "<unset>")]
    if not changed:
        print("config unchanged")
    for k in changed:
        print(f"  {k}: {json.dumps(a.get(k, '<unset>'))} -> {json.dumps(b.get(k, '<unset>'))}")


def latest_checkpoint_step(folder, name):
    steps = []
    try:
        for f in os.listdir(folder):
            m = re.fullmatch(rf"{re.escape(name)}_(\d+)\.safetensors", f)
            if m:
                steps.append(int(m.group(1)))
    except OSError:
        pass
    return max(steps, default=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-u", "--update", action="store_true", help="update the job if it already exists")
    parser.add_argument("--gpus", help="GPU ids, e.g. '0' or '0,1' (default: keep existing on update, "
                        "else inferred from ddp_devices in the config, else '0')")
    parser.add_argument("name", help="job name")
    parser.add_argument("config", help="config file (.yaml/.yml/.json)")
    args = parser.parse_args()

    try:
        cfg = load_config(os.path.expanduser(args.config))
        proc = cfg["config"]["process"][0]
    except (OSError, ValueError, yaml.YAMLError) as e:
        sys.exit(f"error: cannot read {args.config}: {e}")
    except (KeyError, IndexError, TypeError):
        sys.exit(f"error: {args.config} has no config.process[0]; not an ai-toolkit job config")

    try:
        settings = aitk_ui.get_settings()
        existing = aitk_ui.find_job(args.name)
    except aitk_ui.UIError as e:
        sys.exit(f"error: {e}")

    if existing and not args.update:
        sys.exit(f"error: job {args.name!r} already exists (status {existing['status']}); use -u to update it")
    if existing and existing["status"] in ACTIVE:
        sys.exit(f"error: job {args.name!r} is {existing['status']}; stop it before updating")
    if existing and existing["job_type"] != "train":
        sys.exit(f"error: {args.name!r} is a {existing['job_type']} job, not a training job")

    # gpu_ids is a job column, not part of the config: the worker regenerates
    # ddp_devices from it at launch, so a stale copy here would only mislead.
    ddp_devices = proc.pop("ddp_devices", None)
    if args.gpus:
        gpu_ids = args.gpus
    elif existing:
        gpu_ids = existing["gpu_ids"]
    elif ddp_devices:
        gpu_ids = ",".join(str(i) for i in range(len(ddp_devices)))
    else:
        gpu_ids = "0"

    # the fields the UI forces on every imported/edited config
    cfg["config"]["name"] = args.name
    proc["sqlite_db_path"] = "./aitk_db.db"
    proc["training_folder"] = settings["TRAINING_FOLDER"]
    proc["device"] = "cuda"
    proc["performance_log_every"] = 10
    # UI jobs log to the job folder so the loss graph works
    proc.setdefault("logging", {})["use_ui_logger"] = True
    cfg.setdefault("meta", {})["name"] = "[name]"

    for ds in proc.get("datasets", []):
        if ds.get("folder_path") and not os.path.isdir(ds["folder_path"]):
            print(f"warning: dataset folder {ds['folder_path']} does not exist")

    job_folder = os.path.join(settings["TRAINING_FOLDER"], args.name)
    last = latest_checkpoint_step(job_folder, args.name)
    steps = proc.get("train", {}).get("steps")
    if last is not None:
        print(f"note: {job_folder} has checkpoints up to step {last}; the job will resume from there")
        if steps is not None and steps <= last:
            print(f"warning: train.steps={steps} <= {last}, so the job would finish immediately")

    if existing:
        print_diff(json.loads(existing["job_config"]), cfg)
        job = aitk_ui.save_job(id=existing["id"], name=args.name, gpu_ids=gpu_ids, job_config=cfg)
        print(f"updated job {job['name']!r} ({job['id']}) gpus={job['gpu_ids']}")
    else:
        try:
            job = aitk_ui.save_job(name=args.name, gpu_ids=gpu_ids, job_config=cfg)
        except aitk_ui.UIError as e:
            sys.exit(f"error: {e}")
        print(f"created job {job['name']!r} ({job['id']}) gpus={job['gpu_ids']} status={job['status']}")
    print(f"{aitk_ui.base_url()}/jobs/{job['id']}")


if __name__ == "__main__":
    main()
