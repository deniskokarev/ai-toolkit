"""Run training stages through the AI Toolkit UI and wait for them to finish.

    train_stage.py [options] <job_name> <config.yaml> [<config.yaml>...]
    train_stage.py <job_name> <config.yaml> <gpus>          (legacy form)

Each config is one stage of the same job (typically the same run with a higher
train.steps and a different lr). For every stage, in order: import/update the
job from the config, queue it, wait for it to finish. Exits 0 only if every
stage ends "completed":

    train_stage.py anm_ct_dd_s_anm anm_lora_step3.yaml anm_lora_step4.yaml

Safe to re-run after ^C or a failure. ^C only stops the waiting; the job keeps
training in the UI. On re-run each stage is identified by its train.steps:
  - checkpoints already at or past train.steps: stage done, skipped
  - job still queued/running this stage: re-attach and wait
  - job queued/running a different stage: error (stop it in the UI first)
  - otherwise (stopped, error, never run): update + start; the trainer
    resumes from the latest checkpoint, never from step 0

Exit status: 0 all stages completed, 1 a stage failed (or the UI went away),
2 usage, 3 stalled (no progress for --stall minutes; the job was stopped),
130 interrupted.

Progress goes to stdout: one self-overwriting line on a terminal, otherwise one
"[progress] ..." line per change (for logs and scripts/remote/aitk_remote.py).

Environment: AITK_UI_URL / AI_TOOLKIT_AUTH as for scripts/aitk_ui.py; POLL
(seconds between status checks, default 30) and STALL (minutes, default off)
are defaults for --poll and --stall.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aitk_ui  # noqa: E402
import import_job  # noqa: E402

ACTIVE = ("queued", "running", "stopping")
GPUS_RE = re.compile(r"^\d+(,\d+)*$")
EXIT_FAIL, EXIT_USAGE, EXIT_STALL = 1, 2, 3


class Stage(Exception):
    """A stage-level failure: message for the user and the exit status."""

    def __init__(self, msg, code=EXIT_FAIL):
        super().__init__(msg)
        self.code = code


def fmt_dur(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


class Progress:
    def __init__(self):
        self.tty = sys.stdout.isatty()
        self.last = None
        self.open = False

    def show(self, text):
        if self.tty:
            print(f"\r{text}\033[K", end="", flush=True)
            self.open = True
        elif text != self.last:
            print(f"[progress] {text}", flush=True)
        self.last = text

    def say(self, text, err=False):
        if self.open:
            print(flush=True)
            self.open = False
        print(text, file=sys.stderr if err else sys.stdout, flush=True)


def log_size(folder):
    try:
        return os.path.getsize(os.path.join(folder, "log.txt"))
    except OSError:
        return -1


def run_stage(args, idx, cfg_path, out):
    tag = f"[{idx + 1}/{len(args.configs)} {os.path.basename(cfg_path)}]"
    try:
        cfg = import_job.load_config(os.path.expanduser(cfg_path))
        target = int(cfg["config"]["process"][0]["train"]["steps"])
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        raise Stage(f"cannot read train.steps from {cfg_path}: {e}")

    settings = aitk_ui.get_settings()
    folder = os.path.join(settings["TRAINING_FOLDER"], args.job)
    done = import_job.latest_checkpoint_step(folder, args.job) or 0

    job = aitk_ui.find_job(args.job)
    if job and job["status"] in ACTIVE:
        running = json.loads(job["job_config"])["config"]["process"][0]["train"]["steps"]
        if int(running) != target:
            raise Stage(f"{args.job} is {job['status']} with train.steps={running}, not this stage "
                        f"({target}); stop it in the UI first")
        out.say(f"{tag} {args.job} is already {job['status']} toward step {target}; waiting — "
                f"{aitk_ui.base_url()}/jobs/{job['id']}")
    else:
        if done >= target:
            out.say(f"{tag} checkpoints at step {done} >= {target}, stage already done; skipping")
            return
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_job.py"),
               "-u", "--gpus", args.gpus]
        for m in args.map_path:
            cmd += ["--map-path", m]
        cmd += [args.job, cfg_path]
        sys.stdout.flush()
        if subprocess.call(cmd) != 0:
            raise Stage(f"importing {cfg_path} failed")
        job = aitk_ui.find_job(args.job)
        if not job:
            raise Stage(f"job {args.job} not found in the UI after import")
        aitk_ui.start_job(job["id"], args.gpus)
        out.say(f"{tag} queued {args.job} ({job['id']}) on gpus {args.gpus}, step {done} -> {target} — "
                f"{aitk_ui.base_url()}/jobs/{job['id']}")

    job_id = job["id"]
    fails = 0
    sig, sig_t = None, time.time()   # last observed progress signature and when it changed
    first_step, rate_from = None, None   # step at attach; (time, step) once it first moved
    last_step, last_step_t = None, None  # for s/it: time of the latest step change, not "now"
    while True:
        time.sleep(args.poll)
        try:
            j = aitk_ui.get_job(job_id)
        except aitk_ui.UIError:
            # tolerate a UI restart or a slow response rather than abandoning the wait
            fails += 1
            if fails >= 20:
                raise Stage(f"UI unreachable for {fails * args.poll}s")
            continue
        fails = 0
        status, step, info = j["status"], int(j.get("step") or 0), j.get("info") or ""
        if status not in ACTIVE:
            break
        now = time.time()
        # the log grows while the model downloads / latents cache, so that counts as progress too
        cur = (status, step, info, log_size(folder))
        if cur != sig:
            sig, sig_t = cur, now
        # rate baseline: the first step change we see, so model loading and
        # latent caching before step 1 do not count
        if first_step is None or step < first_step:
            first_step, rate_from = step, None
        elif rate_from is None and step != first_step:
            rate_from = (now, step)
        if step != last_step:
            last_step, last_step_t = step, now
        eta = ""
        if rate_from and step > rate_from[1]:
            spi = (last_step_t - rate_from[0]) / (step - rate_from[1])
            eta = f"  {spi:.2f}s/it  eta {fmt_dur(spi * max(target - step, 0))}"
        out.show(f"{tag} {time.strftime('%H:%M')}  {status:<8} step {step}/{target}{eta}  {info}")
        if args.stall and now - sig_t > args.stall * 60:
            out.say(f"{tag} no progress for {args.stall} min (status {status}, step {step}, {info!r}); "
                    f"stopping {args.job}", err=True)
            try:
                aitk_ui.stop_job(job_id)
            except aitk_ui.UIError as e:
                out.say(f"warning: could not stop the job: {e}", err=True)
            raise Stage("stalled", EXIT_STALL)

    out.say(f"{tag} {args.job} finished: {status} at step {step} — {info}")
    if status != "completed":
        raise Stage(f"stage {cfg_path} ended {status}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    p.add_argument("-g", "--gpus", help="GPU ids for the job, e.g. 0 or 0,1 (default: every GPU the UI sees)")
    p.add_argument("--map-path", metavar="OLD=NEW", action="append", default=[],
                   help="passed to import_job.py: rewrite config path prefix OLD to NEW (repeatable)")
    p.add_argument("--stall", type=float, default=float(os.environ.get("STALL") or 0), metavar="MIN",
                   help="stop the job and exit 3 after MIN minutes without progress (default off)")
    p.add_argument("--poll", type=float, default=float(os.environ.get("POLL") or 30), metavar="SEC",
                   help="seconds between status checks (default 30)")
    p.add_argument("job")
    p.add_argument("configs", nargs="+", metavar="config")
    args = p.parse_args()

    # legacy: train_stage.sh <job> <config> <gpus>
    if (len(args.configs) == 2 and not args.gpus and GPUS_RE.match(args.configs[1])
            and not os.path.exists(args.configs[1])):
        args.gpus = args.configs.pop()
    for c in args.configs:
        if not os.path.isfile(os.path.expanduser(c)):
            p.error(f"no such config: {c}")
    if args.gpus and not GPUS_RE.match(args.gpus):
        p.error(f"bad --gpus {args.gpus!r}")

    out = Progress()

    def on_int(*_):
        out.say(f"stopped waiting; {args.job} keeps training in the UI. Re-run to re-attach.", err=True)
        sys.exit(130)

    signal.signal(signal.SIGINT, on_int)
    try:
        if not args.gpus:
            n = aitk_ui.gpu_count() or 1
            args.gpus = ",".join(str(i) for i in range(n))
        for i, c in enumerate(args.configs):
            run_stage(args, i, c, out)
    except Stage as e:
        out.say(f"error: {e}", err=True)
        sys.exit(e.code)
    except aitk_ui.UIError as e:
        out.say(f"error: {e}", err=True)
        sys.exit(EXIT_FAIL)
    out.say(f"{args.job}: all {len(args.configs)} stage(s) completed")


if __name__ == "__main__":
    main()
