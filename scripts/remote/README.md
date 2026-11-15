# Remote (RunPod) batch training

`aitk_remote.py` runs a staged ai-toolkit job on a remote box: a RunPod pod
from the `ostris/aitoolkit` image, or any host you can ssh to (including your
own machine over loopback). It runs locally, needs only Python 3 (stdlib),
`ssh`, `rsync`, and `runpodctl` for pods. The box needs `git`, `rsync`, and
`setsid` (util-linux). The docker image has all three.

```
scripts/remote/aitk_remote.py run anm_x stage1.yaml stage2.yaml --dest ~/loras --terminate
```

`run` does:

1. **Find the pod.** It uses `--pod ID|NAME`, otherwise the single *running*
   pod whose name starts with `ai-toolkit`. Zero or several matches abort with
   a list. ssh details come from `runpodctl pod get`, and it waits for a
   starting pod's sshd. The host key is accepted the first time it's seen
   (`accept-new`) and stored in a known_hosts file for that pod alone, under
   `~/.cache/aitk_remote/`, so a later pod on the same recycled ip:port doesn't
   trip the "host key changed" check. ssh connections are multiplexed
   (ControlMaster).
2. **Update git.** In `/app/ai-toolkit`, `origin` becomes your fork (by
   default this checkout's `origin`, rewritten as https; the image's ostris
   remote becomes `upstream`), and the checkout switches to `--branch`
   (default `batch_training`). If `ui/` changed since the UI was last built,
   it rebuilds the UI and restarts the container (see below).
3. **Upload datasets.** It rsyncs every `--dataset DIR[=NAME]`, plus every
   dataset folder the configs name that exists locally, to the box's
   `DATASETS_FOLDER/NAME`. Trainer caches (`_latent_cache` etc.) are skipped.
   Your configs stay as they are: their local paths are rewritten to the
   uploaded copies when the job is imported (`import_job.py --map-path`).
   Symlinked paths are matched by realpath.
4. **Launch.** It starts `scripts/train_stage.sh JOB CONFIG...` detached
   (`setsid -f`), so it survives ssh drops and this program exiting. The
   configs, log (`run.log`), pid and exit status go to
   `<TRAINING_FOLDER>/JOB/.batch/` on the box, which also syncs back to you.
5. **Watch.** It streams the log and keeps a live progress line, e.g.
   `[2/2 s2.yaml] 14:02 running step 412/6000 1.83s/it eta 2:50:24`. It
   rsyncs `<TRAINING_FOLDER>/JOB/` to `DEST/JOB/` every `--sync-every`
   minutes (default 10) and once more at the end. ^C only stops watching.
   Re-attach with `watch JOB`.
6. **End the pod (optional).** `--terminate` deletes the pod, and `--stop`
   stops it (the volume is kept and still billed). Either happens once the
   batch is over and the final sync succeeded, whatever the outcome:
   completed, failed, or stalled. With either flag, `--stall` defaults to 30
   minutes: a job with no progress (no step, status or info change, and no
   log growth) for that long is stopped and counts as over. If the final sync
   fails, the pod is left alone.

   This needs the local program to still be watching at the end. Run it in
   tmux/screen, or re-attach later with `watch JOB --terminate`.

Other commands:

| command | what it does |
|---|---|
| `watch JOB [--terminate]` | re-attach: log tail, progress, periodic sync |
| `sync JOB [--dest D]` | one rsync of the job folder now |
| `stop JOB` | kill the remote batch and stop the UI job |
| `setup` | step 2 only |
| `where` | print the resolved target and its ssh command |
| `ssh [CMD]` | shell (in the checkout) or a command on the target |

Models given as Hugging Face ids download on the box using the pod's
`HF_TOKEN`. The UI worker that launches training runs with the container's
environment.

## Staged training: `scripts/train_stage.sh`

```
scripts/train_stage.sh [-g GPUS] [--stall MIN] JOB stage1.yaml stage2.yaml ...
scripts/train_stage.sh JOB stage.yaml 0,1        # the old one-stage form
```

This is a port of `nlora_tools/train_stage.sh` with these changes:

- it takes several stages
- `-g` defaults to every GPU the UI sees, not `0,1`
- it finds the UI on :3000 (dev/systemd) or :8675 (docker image)
- it picks up the token from `/proc/1/environ` inside a container
- it adds an s/it and ETA readout
- `--stall` stall detection, exit 3
- plain `[progress]` lines when stdout isn't a terminal
- no `jq`/`curl` dependency, since the image has no `jq`

Re-running is safe, with the same stage logic as before: stages whose
checkpoints are already at their `train.steps` are skipped, and a running
stage is re-attached.

## The UI rebuild and container restart

The image ships a UI built from upstream; this branch changes `ui/` (worker
multi-GPU launch, ROCm stats). After a rebuild, `pod_setup.sh` SIGKILLs the
image's `concurrently` supervisor. That is the container's main process, so
the container restarts. The disk survives (checkout and new build), the UI
comes back on the new build with the container's own env, and
`aitk_remote.py` reconnects and waits for it. The build commit is recorded in
`ui/dist/.aitk_built_rev`, so this happens only when `ui/` actually changes.
It refuses while any job is active.

## Local / loopback use

```
scripts/remote/aitk_remote.py --host localhost --root ~/ai-toolkit run JOB s1.yaml s2.yaml --dest /tmp/x
```

With `--host`, the defaults are careful. The checkout's remotes are left
alone (pass `--repo URL` to change `origin`). Git runs in `ff` mode: it never
resets, refuses uncommitted changes, and only fast-forwards `--branch`, or
moves off `main` onto it. It refuses outright when the checkout is on any
other branch, so pass `--no-git` there. The UI is never rebuilt
(`--rebuild-ui never`), and datasets already in place aren't copied.
`--terminate` is ignored.

## Developing these tools (the chicken-and-egg problem)

The pod has to run code you haven't pushed yet. `--push-local` pushes your
local `HEAD` straight into the pod's checkout over the same ssh connection
(`refs/remotes/dev/<branch>`) and checks that out, with no GitHub round trip.
Commit (a WIP commit is fine), then:

```
scripts/remote/aitk_remote.py setup --push-local      # or run ... --push-local
```

Once the branch is on your fork, drop `--push-local` and the pod fetches
`origin/<branch>`.

## TODO

- **Create the pod as well** (`create` subcommand, then `run` against it).
  This is how a pod is created today:

  ```
  runpodctl pod create \
    --cloud-type SECURE \
    --container-disk-in-gb 200 \
    --volume-in-gb 200 \
    --volume-mount-path /workspace \
    --gpu-id "NVIDIA RTX PRO 6000 Blackwell Server Edition" \
    --gpu-count 1 \
    --image "ostris/aitoolkit:latest" \
    --ports "22/tcp,443/http,8080/tcp,8188/http,8888/http,8675/http" \
    --env '{"PUBLIC_KEY":"<ssh public key>","HF_TOKEN":"<hf token>","AI_TOOLKIT_AUTH":"<ui password>"}' \
    --name ai-toolkit-<something>
  ```

  Plan: `create [--gpu-id ...] [--gpu-count N] [--name ...]` reads
  `HF_TOKEN`/`AI_TOOLKIT_AUTH` from the environment or
  `~/.config/ai-toolkit.env` and `PUBLIC_KEY` from `~/.ssh/*.pub`, so secrets
  never sit in shell history. It prints the pod id and lets `run --pod ID`
  take it from there; `resolve()` already waits for a starting pod's ssh. The
  name must start with `ai-toolkit` for auto-discovery. The models then
  download into the 200 GB container disk. Consider pointing `HF_HOME` at
  `/workspace` so a stopped and restarted pod keeps them.
- Multi-GPU pods: `-g` already defaults to all GPUs, but this hasn't been
  tested on a multi-GPU pod yet.
