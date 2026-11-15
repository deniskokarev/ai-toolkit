#!/usr/bin/env python3
"""Train an ai-toolkit job on a remote box (a RunPod pod, or any ssh host).

    aitk_remote.py run   JOB CONFIG... [--dataset DIR[=NAME]]... [--terminate]
    aitk_remote.py watch JOB [--terminate]     re-attach after a disconnect
    aitk_remote.py sync  JOB                   one rsync of the job's output now
    aitk_remote.py stop  JOB                   stop the remote batch and the job
    aitk_remote.py setup                       git update (+ UI rebuild) only
    aitk_remote.py where                       print the resolved target
    aitk_remote.py ssh [CMD...]                shell on the target

`run` does, in order:
  1. target: --host, or --pod ID|NAME, or the one running RunPod pod named
     ai-toolkit* (runpodctl); several or none: abort. Its ssh host key is
     accepted on first sight, into a known_hosts file of that pod's own.
  2. git: make the remote checkout (/app/ai-toolkit in the image) track your
     fork as `origin` and check out --branch (default batch_training), then
     rebuild/restart the UI if ui/ changed (scripts/remote/pod_setup.sh).
     --push-local instead pushes your local HEAD there over ssh, so unpushed
     commits can be tried without GitHub.
  3. datasets: rsync every --dataset DIR, and every dataset folder the configs
     name that exists here, into the remote DATASETS_FOLDER; the configs' paths
     are rewritten to the uploaded copies at import time (--map-path).
  4. launch scripts/train_stage.sh JOB CONFIG... detached on the remote
     (setsid -f: survives ssh drops and this program exiting); its log and
     exit status go to <TRAINING_FOLDER>/JOB/.batch/.
  5. watch: print its progress, rsync <TRAINING_FOLDER>/JOB into --dest/JOB
     every --sync-every minutes and once more at the end.
  6. --terminate (--stop): once the batch is over -- completed, failed, or
     stalled (no progress for --stall minutes, default 30 then) -- and the final
     sync succeeded, delete (stop) the pod. Needs this program still watching:
     run it inside tmux/screen or re-attach with `watch JOB --terminate`.

Models named by Hugging Face id download on the box with the pod's HF_TOKEN.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_REPO = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "aitk_remote")
POD_ROOT = "/app/ai-toolkit"
POD_PREFIX = "ai-toolkit"
# rsync anchored excludes: the trainer's caches inside dataset folders are
# rebuilt on the box (they depend on the model/arch anyway)
DATASET_EXCLUDES = ["--exclude=/_*cache*/", "--exclude=/.aitk*"]

q = shlex.quote


def die(msg, code=1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def note(msg):
    print(f"aitk_remote: {msg}", flush=True)


# --------------------------------------------------------------------------- target

class Target:
    def __init__(self, host, port=None, user=None, key=None, pod_id=None, root=None):
        self.host, self.port, self.user, self.key, self.pod_id = host, port, user, key, pod_id
        self.root = root
        self.is_pod = pod_id is not None
        os.makedirs(CACHE, exist_ok=True)

    @property
    def dest(self):
        return f"{self.user}@{self.host}" if self.user else self.host

    def ssh_opts(self, with_port=True):
        o = ["-x", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "LogLevel=ERROR",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
             "-o", "ControlMaster=auto", "-o", f"ControlPath={CACHE}/cm-%C", "-o", "ControlPersist=10m",
             "-o", "StrictHostKeyChecking=accept-new"]
        if self.is_pod:
            # pods recycle ip:port, so one known_hosts per pod: accept-new pins
            # the key the first time without clashing with an older pod's
            o += ["-o", f"UserKnownHostsFile={CACHE}/known_hosts_{self.pod_id}"]
        if self.key:
            o += ["-i", self.key]
        if with_port and self.port:
            o += ["-p", str(self.port)]
        return o

    def ssh_cmd(self, *extra):
        return ["ssh", *self.ssh_opts(), *extra, self.dest]

    def rsh(self):
        return " ".join(q(a) for a in ["ssh", *self.ssh_opts()])

    def run(self, script, check=True, capture=False, stdin=None, timeout=None):
        """Run a bash script on the target. Returns CompletedProcess."""
        cmd = self.ssh_cmd() + ["bash -c " + q(script)]
        r = subprocess.run(cmd, input=stdin, text=True, timeout=timeout,
                           stdout=subprocess.PIPE if capture else None,
                           stderr=subprocess.PIPE if capture else None)
        if check and r.returncode != 0:
            err = (r.stderr or "").strip() if capture else ""
            die(f"remote command failed (exit {r.returncode}){': ' + err if err else ''}")
        return r

    def out(self, script, **kw):
        return self.run(script, capture=True, **kw).stdout

    def git_url(self):
        port = f":{self.port}" if self.port else ""
        return f"ssh://{self.dest}{port}{self.root}"

    def __str__(self):
        s = f"{self.dest}" + (f" port {self.port}" if self.port else "")
        return (f"pod {self.pod_id} ({s})" if self.is_pod else s) + f", root {self.root}"


def runpodctl(*args):
    if not shutil.which("runpodctl"):
        die("runpodctl not found; install it or pass --host")
    r = subprocess.run(["runpodctl", *args, "-o", "json"], capture_output=True, text=True)
    if r.returncode != 0:
        die(f"runpodctl {' '.join(args)} failed: {r.stderr.strip() or r.stdout.strip()}")
    return json.loads(r.stdout or "null")


def find_pod(spec):
    pods = runpodctl("pod", "list") or []
    if spec:
        hits = [p for p in pods if spec in (p.get("id"), p.get("name"))]
    else:
        hits = [p for p in pods if (p.get("name") or "").startswith(POD_PREFIX)
                and p.get("desiredStatus") == "RUNNING"]
    if len(hits) != 1:
        listing = "\n".join(f"  {p.get('id')}  {p.get('name')}  {p.get('desiredStatus')}" for p in pods) or "  (none)"
        what = f"matching {spec!r}" if spec else f"running and named {POD_PREFIX}*"
        die(f"{len(hits)} pods {what}; pass --pod ID. Pods:\n{listing}")
    return hits[0]["id"]


def resolve(args):
    if args.host:
        m = re.fullmatch(r"(?:([^@]+)@)?([^:]+)(?::(\d+))?", args.host)
        if not m:
            die(f"bad --host {args.host!r}; expected [user@]host[:port]")
        t = Target(m.group(2), m.group(3), m.group(1), args.identity, None, args.root)
        if not t.root:
            die("--root is required with --host (the ai-toolkit checkout there)")
    else:
        pod_id = find_pod(args.pod)
        deadline = time.time() + args.wait_ssh
        while True:
            pod = runpodctl("pod", "get", pod_id)
            ssh = pod.get("ssh") or {}
            if ssh.get("ip") and ssh.get("port"):
                break
            if time.time() > deadline:
                die(f"pod {pod_id} has no ssh endpoint after {args.wait_ssh}s "
                    f"(status {pod.get('desiredStatus')}/{pod.get('runtimeStatus')})")
            note(f"waiting for pod {pod_id} ssh ({pod.get('runtimeStatus') or pod.get('desiredStatus')})")
            time.sleep(10)
        key = args.identity
        if not key and (ssh.get("ssh_key") or {}).get("exists"):
            key = ssh["ssh_key"].get("path")
        t = Target(ssh["ip"], ssh["port"], "root", key, pod_id, args.root or POD_ROOT)
    # sshd comes up a bit after the pod reports its port
    deadline = time.time() + args.wait_ssh
    while True:
        r = subprocess.run(t.ssh_cmd() + ["true"], capture_output=True, text=True)
        if r.returncode == 0:
            break
        if time.time() > deadline:
            die(f"cannot ssh to {t}: {r.stderr.strip()}")
        note(f"waiting for ssh on {t.dest}: {r.stderr.strip()[:120]}")
        time.sleep(10)
    if t.root.startswith("~"):
        t.root = t.out('echo "$HOME"').strip() + t.root[1:]
    return t


# --------------------------------------------------------------------------- remote helpers

def remote_py(t, code):
    """Run python with the remote toolkit's scripts/ importable; returns stdout."""
    script = (f"cd {q(t.root)}/scripts && PY=../venv/bin/python; [ -x $PY ] || PY=$(command -v python3); "
              f"exec $PY -c {q(code)}")
    return t.out(script)


def remote_settings(t):
    return json.loads(remote_py(t, "import aitk_ui, json; print(json.dumps(aitk_ui.get_settings()))"))


def batch_dir(t, job):
    tf = remote_settings(t)["TRAINING_FOLDER"]
    return tf, f"{tf.rstrip('/')}/{job}", f"{tf.rstrip('/')}/{job}/.batch"


STATE_SH = r'''B=%s
if [ -f "$B/rc" ]; then st="done $(cat "$B/rc")"
elif [ -f "$B/pid" ] && kill -0 "$(cat "$B/pid")" 2>/dev/null; then st=running
elif [ -f "$B/pid" ]; then st=dead
else st=none; fi
s=$(wc -c 2>/dev/null <"$B/run.log" || echo 0); s=${s// /}
echo "$st $s"
'''


def remote_state(t, bdir, timeout=60):
    """('running'|'done'|'dead'|'none', rc or None, log size)"""
    out = t.out(STATE_SH % q(bdir), timeout=timeout).split()
    rc = int(out[1]) if out[0] == "done" else None
    return out[0], rc, int(out[-1])


# --------------------------------------------------------------------------- steps

def default_repo():
    try:
        url = subprocess.run(["git", "-C", LOCAL_REPO, "remote", "get-url", "origin"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return ""
    m = re.fullmatch(r"git@github\.com:(.+?)(?:\.git)?", url)
    return f"https://github.com/{m.group(1)}.git" if m else url


def step_git(t, args):
    if args.no_git:
        note("--no-git: leaving the remote checkout as it is")
        return
    source = "origin"
    if args.push_local:
        dirty = subprocess.run(["git", "-C", LOCAL_REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            note("warning: local uncommitted changes are NOT pushed (only HEAD is)")
        env = dict(os.environ, GIT_SSH_COMMAND=" ".join(q(a) for a in ["ssh", *t.ssh_opts(with_port=False)]))
        note(f"pushing local HEAD to {t.dest}:{t.root} as dev/{args.branch}")
        r = subprocess.run(["git", "-C", LOCAL_REPO, "push", "-q", "-f", t.git_url(),
                            f"HEAD:refs/remotes/dev/{args.branch}"], env=env)
        if r.returncode != 0:
            die("git push to the target failed")
        source = "dev"
    mode = args.git_mode or ("reset" if t.is_pod else "ff")
    rebuild = args.rebuild_ui or ("auto" if t.is_pod else "never")
    # a pod's checkout gets your fork as origin; a --host checkout is yours,
    # its remotes are left alone unless --repo says otherwise
    repo = args.repo if args.repo is not None else (default_repo() if t.is_pod else "")
    with open(os.path.join(HERE, "pod_setup.sh")) as f:
        script = f.read()
    cmd = t.ssh_cmd() + ["bash -s -- " + " ".join(q(a) for a in (t.root, args.branch, source, mode, repo, rebuild))]
    boot = container_start(t)
    rc = subprocess.run(cmd, input=script, text=True).returncode
    if rc == 75:
        wait_restart(t, boot)
    elif rc != 0:
        die("remote git/UI setup failed")


def container_start(t):
    """Start time of the box's PID 1: changes when the container restarts."""
    return t.out("ps -o lstart= -p 1", check=False).strip()


def wait_restart(t, boot, timeout=600):
    """The container is restarting (new UI build): reconnect, wait for the UI."""
    drop_mux = ["ssh", *t.ssh_opts(), "-O", "exit", t.dest]
    subprocess.run(drop_mux, capture_output=True)
    time.sleep(15)
    probe = (f"cd {q(t.root)}/scripts && PY=../venv/bin/python; [ -x $PY ] || PY=$(command -v python3); "
             "$PY -c 'import aitk_ui; aitk_ui.get_settings()'")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if container_start(t) not in ("", boot) and \
                    t.run(probe, check=False, capture=True, timeout=60).returncode == 0:
                note("container restarted and the UI answers")
                return
        except subprocess.TimeoutExpired:
            pass
        subprocess.run(drop_mux, capture_output=True)
        note("waiting for the container to come back")
        time.sleep(10)
    die(f"the target did not come back within {timeout}s after the UI restart")


# config keys import_job.py overwrites anyway: not worth a warning
IGNORED_KEYS = ("training_folder", "sqlite_db_path")


def walk_strings(obj, key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_strings(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v, key)
    elif isinstance(obj, str) and key not in IGNORED_KEYS:
        yield obj


def config_paths(path):
    """Absolute path strings in a config, plus the dataset-ish ones."""
    with open(path) as f:
        text = f.read()
    try:
        if path.endswith((".json", ".jsonc")):
            cfg = json.loads(text)
        else:
            import yaml  # not in every local python; fall back to a line scan
            cfg = yaml.safe_load(text)
        strings = list(walk_strings(cfg))
        datasets = []
        for ds in cfg["config"]["process"][0].get("datasets") or []:
            datasets += [v for k, v in ds.items() if k.endswith("_path") and isinstance(v, str)]
    except ImportError:
        pairs = re.findall(r"^[\s-]*([\w]+)\s*:\s*['\"]?(/[^'\"#\n]*?)['\"]?\s*$", text, re.M)
        strings = [v for k, v in pairs if k not in IGNORED_KEYS]
        datasets = [v for k, v in pairs if k.endswith("_path") and k not in IGNORED_KEYS]
    return [s for s in strings if s.startswith("/")], [d for d in datasets if d.startswith("/")]


def step_datasets(t, args, settings):
    """Upload datasets; return --map-path list (literal config path -> remote path)."""
    uploads = {}  # realpath -> name
    for spec in args.dataset:
        d, _, name = spec.partition("=")
        if not os.path.isdir(d):
            die(f"--dataset {d}: not a directory")
        uploads[os.path.realpath(d)] = name or os.path.basename(os.path.realpath(d))
    literal = set()
    for c in args.configs:
        strings, datasets = config_paths(c)
        literal.update(strings)
        if not args.no_auto_datasets:
            for d in datasets:
                if os.path.isdir(d):
                    uploads.setdefault(os.path.realpath(d), os.path.basename(os.path.realpath(d)))
                else:
                    note(f"warning: {c}: dataset {d} is not here; assuming it already exists on the target")
    names = {}
    for real, name in uploads.items():
        if name in names:
            die(f"two datasets would both upload as {name!r}: {names[name]} and {real}; "
                f"name one with --dataset DIR=NAME")
        names[name] = real

    rds = settings["DATASETS_FOLDER"].rstrip("/")
    maps = {}
    for p in sorted(literal):
        rp = os.path.realpath(p)
        for real, name in uploads.items():
            if rp == real or rp.startswith(real + "/"):
                maps[p] = f"{rds}/{name}{rp[len(real):]}"
                break
        else:
            if os.path.exists(p) and not (t.is_pod is False and p.startswith(t.root)):
                note(f"warning: config path {p} exists here but is not uploaded; the target may not have it")
    if uploads:
        t.run(f"mkdir -p {q(rds)}")
    for real, name in sorted(uploads.items(), key=lambda kv: kv[1]):
        dst = f"{rds}/{name}"
        if not t.is_pod and os.path.realpath(dst) == real:
            continue  # loopback: already in place
        note(f"uploading dataset {real} -> {dst}")
        r = subprocess.run(["rsync", "-az", "--partial", *DATASET_EXCLUDES, "-e", t.rsh(),
                            real + "/", f"{t.dest}:{dst}/"])
        if r.returncode != 0:
            die(f"rsync of {real} failed (exit {r.returncode})")
    for old, new in maps.items():
        if old != new:
            note(f"config path {old} -> {new}")
    return [f"{o}={n}" for o, n in maps.items() if o != n]


def launch(t, args, settings, maps):
    tf = settings["TRAINING_FOLDER"].rstrip("/")
    bdir = f"{tf}/{args.job}/.batch"
    st, _, _ = remote_state(t, bdir)
    if st == "running":
        die(f"a batch for {args.job} is already running there; `watch {args.job}` to follow it "
            f"or `stop {args.job}` first")
    t.run(f"mkdir -p {q(bdir)}/cfg && rm -f {q(bdir)}/cfg/* {q(bdir)}/rc {q(bdir)}/pid")
    remote_cfgs = []
    for i, c in enumerate(args.configs):
        remote_cfgs.append(f"{bdir}/cfg/{i + 1:02d}_{os.path.basename(c)}")
    # one rsync per file keeps the numbered names
    for c, rc in zip(args.configs, remote_cfgs):
        r = subprocess.run(["rsync", "-a", "-e", t.rsh(), c, f"{t.dest}:{rc}"])
        if r.returncode != 0:
            die(f"uploading {c} failed")
    ts = [f"{t.root}/scripts/train_stage.sh"]
    if args.gpus:
        ts += ["-g", args.gpus]
    if args.stall:
        ts += ["--stall", str(args.stall)]
    for m in maps:
        ts += ["--map-path", m]
    ts += [args.job, *remote_cfgs]
    inner = (f"echo $$ > {q(bdir)}/pid; echo \"=== $(date -u '+%F %T') UTC batch start\"; "
             f"{' '.join(q(a) for a in ts)}; rc=$?; "
             f"echo \"=== $(date -u '+%F %T') UTC batch end, exit $rc\"; echo $rc > {q(bdir)}/rc")
    # setsid -f: own session, forked off so nothing waits on it; ssh drops,
    # SIGHUP and this program exiting do not touch it
    script = (f"cd {q(t.root)} && PYTHONUNBUFFERED=1 setsid -f bash -c {q(inner)} "
              f">>{q(bdir)}/run.log 2>&1 </dev/null")
    t.run(script)
    note(f"launched on {t.dest}; log {bdir}/run.log")
    return bdir


def rsync_down(t, job, settings_tf, dest):
    src = f"{settings_tf.rstrip('/')}/{job}/"
    local = os.path.join(dest, job)
    if not t.is_pod and os.path.realpath(local) == os.path.realpath(src.rstrip("/")):
        return True  # loopback with dest == the output folder itself
    os.makedirs(local, exist_ok=True)
    r = subprocess.run(["rsync", "-az", "--partial", "-e", t.rsh(), f"{t.dest}:{src}", local + "/"],
                       stdout=subprocess.DEVNULL)
    if r.returncode != 0:
        note(f"warning: rsync of {src} failed (exit {r.returncode}); will retry")
        return False
    return True


class Printer:
    def __init__(self):
        self.tty = sys.stdout.isatty()
        self.open = False
        self.partial = ""

    def feed(self, chunk):
        self.partial += chunk
        *lines, self.partial = self.partial.replace("\r", "\n").split("\n")
        for line in lines:
            self.line(line)

    def line(self, line):
        if line.startswith("[progress] ") and self.tty:
            print(f"\r{line[11:]}\033[K", end="", flush=True)
            self.open = True
            return
        if not line.strip():
            return
        if self.open:
            print()
            self.open = False
        print(line, flush=True)

    def close(self):
        if self.open:
            print()
            self.open = False


def watch(t, args):
    tf = remote_settings(t)["TRAINING_FOLDER"]
    bdir = f"{tf.rstrip('/')}/{args.job}/.batch"
    st, rc, size = remote_state(t, bdir)
    if st == "none":
        die(f"no batch for {args.job} on {t.dest} ({bdir})")
    off = getattr(args, "log_offset", None)
    if off is None:
        # re-attach: replay the tail of the log
        off = max(0, size - 3000)
    pr = Printer()
    every = args.sync_every * 60
    next_sync = time.time() + every
    fails, last_warn = 0, 0.0
    note(f"watching {args.job} on {t.dest}; syncing to {os.path.join(args.dest, args.job)} every "
         f"{args.sync_every:g} min (^C stops watching only)")
    try:
        while True:
            try:
                r = t.run(STATE_SH % q(bdir) + f'[ "$s" -gt {off} ] && tail -c +{off + 1} "$B/run.log" | head -c $((s - {off}))',
                          check=False, capture=True, timeout=120)
                if r.returncode not in (0, 1):
                    raise subprocess.SubprocessError(r.stderr.strip())
                head, _, body = r.stdout.partition("\n")
                parts = head.split()
                st, rc = parts[0], (int(parts[1]) if parts[0] == "done" else None)
                off += len(body.encode())
                pr.feed(body)
                fails = 0
            except (subprocess.SubprocessError, ValueError, IndexError) as e:
                fails += 1
                if time.time() - last_warn > 300:
                    pr.close()
                    note(f"connection problem ({str(e)[:100] or 'no answer'}); retrying")
                    last_warn = time.time()
                if t.is_pod and fails % 10 == 0:
                    pods = runpodctl("pod", "list") or []
                    if not any(p.get("id") == t.pod_id and p.get("desiredStatus") == "RUNNING" for p in pods):
                        pr.close()
                        die(f"pod {t.pod_id} is no longer running")
                time.sleep(args.poll)
                continue
            if st != "running":
                break
            if time.time() >= next_sync:
                rsync_down(t, args.job, tf, args.dest)
                next_sync = time.time() + every
            time.sleep(args.poll)
    except KeyboardInterrupt:
        pr.close()
        note(f"stopped watching; the batch keeps running. Re-attach: watch {args.job}")
        sys.exit(130)
    pr.close()
    if st == "dead":
        note("the remote batch died without an exit status (box restarted or killed?)")
        rc = 1
    note(f"batch over (exit {rc}: {RC_TEXT.get(rc, 'failed')}); final sync")
    synced = any(rsync_down(t, args.job, tf, args.dest) or time.sleep(15) for _ in range(3))
    if not synced:
        die("final sync failed; NOT terminating the pod", 1)
    note(f"synced to {os.path.join(args.dest, args.job)}")
    end_pod(t, args)
    sys.exit(rc)


RC_TEXT = {0: "all stages completed", 3: "stalled", 130: "interrupted", 143: "stopped"}


def end_pod(t, args):
    action = "delete" if args.terminate else "stop" if args.stop_pod else None
    if not action:
        return
    if not t.is_pod:
        note(f"--{'terminate' if args.terminate else 'stop'} ignored: target is not a RunPod pod")
        return
    note(f"{action} pod {t.pod_id}")
    runpodctl("pod", action, t.pod_id)


# --------------------------------------------------------------------------- commands

def cmd_run(t, args):
    for c in args.configs:
        if not os.path.isfile(c):
            die(f"no such config: {c}")
    if args.stall is None:
        args.stall = 30 if (args.terminate or args.stop_pod) else 0
    step_git(t, args)
    settings = remote_settings(t)
    maps = step_datasets(t, args, settings)
    tf = settings["TRAINING_FOLDER"].rstrip("/")
    _, _, size = remote_state(t, f"{tf}/{args.job}/.batch")
    bdir = launch(t, args, settings, maps)
    args.log_offset = size
    watch(t, args)


def cmd_stop(t, args):
    _, _, bdir = batch_dir(t, args.job)
    st, _, _ = remote_state(t, bdir)
    if st == "running":
        # setsid made the batch shell a session/group leader: signal the group
        t.run(f'B={q(bdir)}; kill -TERM -- -"$(cat "$B/pid")" 2>/dev/null; sleep 1; echo 143 > "$B/rc"; '
              f'echo "=== $(date -u "+%F %T") UTC batch stopped by aitk_remote" >> "$B/run.log"')
        note("remote batch stopped")
    else:
        note(f"no running batch ({st})")
    remote_py(t, f"""import aitk_ui
j = aitk_ui.find_job({args.job!r})
if j and j['status'] in ('running', 'queued'):
    aitk_ui.stop_job(j['id']); print('stopped UI job', j['name'])
else:
    print('UI job is', j and j['status'])""")
    sys.stdout.write("")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("\n\n", 1)[1])
    g = p.add_argument_group("target")
    g.add_argument("--pod", help="RunPod pod id or name (default: the one running pod named ai-toolkit*)")
    g.add_argument("--host", help="[user@]host[:port] instead of a pod (e.g. localhost for loopback)")
    g.add_argument("--root", help=f"ai-toolkit checkout on the target (default {POD_ROOT} on a pod)")
    g.add_argument("-i", "--identity", help="ssh private key (default: runpodctl's key for pods, else ssh's)")
    g.add_argument("--wait-ssh", type=int, default=600, metavar="SEC",
                   help="how long to wait for a starting pod's ssh (default 600)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def git_opts(sp):
        sp.add_argument("--branch", default=os.environ.get("AITK_REMOTE_BRANCH", "batch_training"))
        sp.add_argument("--repo", default=os.environ.get("AITK_REMOTE_REPO"),
                        help="URL for the remote's origin (default on a pod: this checkout's origin, as https; "
                             "with --host: leave the remotes alone); '' leaves them")
        sp.add_argument("--push-local", action="store_true",
                        help="push local HEAD to the target over ssh instead of fetching from --repo")
        sp.add_argument("--git-mode", choices=("reset", "ff"),
                        help="reset: hard-reset the branch (default on a pod); ff: fast-forward only (default with --host)")
        sp.add_argument("--rebuild-ui", choices=("auto", "always", "never"),
                        help="rebuild+restart the UI when ui/ changed (default auto on a pod, never with --host)")
        sp.add_argument("--no-git", action="store_true", help="skip the git/UI step")

    def watch_opts(sp):
        sp.add_argument("--dest", default=".", help="local dir; output lands in DEST/JOB (default .)")
        sp.add_argument("--sync-every", type=float, default=10, metavar="MIN", help="rsync period (default 10)")
        sp.add_argument("--poll", type=float, default=30, metavar="SEC", help="status period (default 30)")
        sp.add_argument("--terminate", action="store_true",
                        help="delete the pod when the batch is over and synced (success or not)")
        sp.add_argument("--stop", dest="stop_pod", action="store_true",
                        help="stop (not delete) the pod instead; its volume is kept and billed")

    sp = sub.add_parser("run", help="setup, upload, launch, watch")
    sp.add_argument("job")
    sp.add_argument("configs", nargs="+", metavar="config")
    sp.add_argument("--dataset", action="append", default=[], metavar="DIR[=NAME]",
                    help="upload DIR to the remote datasets folder as NAME (default basename); repeatable")
    sp.add_argument("--no-auto-datasets", action="store_true",
                    help="only upload --dataset dirs, not every dataset folder the configs name")
    sp.add_argument("-g", "--gpus", help="GPU ids for the job (default: all the remote UI sees)")
    sp.add_argument("--stall", type=float, metavar="MIN",
                    help="stop the job when no progress for MIN minutes (default 30 with --terminate/--stop, else off)")
    git_opts(sp)
    watch_opts(sp)

    sp = sub.add_parser("watch", help="follow a running batch, sync, optionally terminate")
    sp.add_argument("job")
    watch_opts(sp)

    sp = sub.add_parser("sync", help="rsync the job's output once")
    sp.add_argument("job")
    sp.add_argument("--dest", default=".")

    sp = sub.add_parser("stop", help="stop the remote batch and its UI job")
    sp.add_argument("job")

    sp = sub.add_parser("setup", help="git update and UI rebuild only")
    git_opts(sp)

    sub.add_parser("where", help="print the resolved target")
    sp = sub.add_parser("ssh", help="ssh to the target")
    sp.add_argument("command", nargs=argparse.REMAINDER)

    args = p.parse_args()
    if args.host and args.pod:
        die("--host and --pod are exclusive", 2)
    t = resolve(args)

    if args.cmd == "where":
        print(t)
        print(" ".join(q(a) for a in t.ssh_cmd()))
    elif args.cmd == "ssh":
        os.execvp("ssh", t.ssh_cmd("-t") + [f"cd {q(t.root)} && exec bash -l"] if not args.command
                  else t.ssh_cmd() + [" ".join(args.command)])
    elif args.cmd == "setup":
        step_git(t, args)
    elif args.cmd == "run":
        cmd_run(t, args)
    elif args.cmd == "watch":
        watch(t, args)
    elif args.cmd == "sync":
        tf = remote_settings(t)["TRAINING_FOLDER"]
        sys.exit(0 if rsync_down(t, args.job, tf, args.dest) else 1)
    elif args.cmd == "stop":
        cmd_stop(t, args)


if __name__ == "__main__":
    main()
