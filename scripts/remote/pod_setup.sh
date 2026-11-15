#!/usr/bin/env bash
# Bring an ai-toolkit checkout to a branch; rebuild/restart the UI if its code
# changed. Runs ON the training box (aitk_remote.py pipes it over ssh):
#
#     pod_setup.sh ROOT BRANCH SOURCE MODE REPO REBUILD
#
#   ROOT     the checkout, e.g. /app/ai-toolkit
#   BRANCH   branch to end up on, e.g. batch_training
#   SOURCE   origin: fetch BRANCH from REPO
#            dev:    use refs/remotes/dev/BRANCH, already pushed here over ssh
#                    (development: lets you run unpushed local commits)
#   MODE     reset:  the box is disposable (a pod): hard-reset BRANCH to SOURCE
#            ff:     a checkout you care about (e.g. loopback): never discard
#                    commits, only fast-forward; refuse if it is on another
#                    branch than BRANCH/main
#   REPO     URL to make `origin` (the image's ostris origin becomes
#            `upstream`); empty: leave remotes alone
#   REBUILD  auto: rebuild the UI if ui/ changed since it was last built
#            always / never
#
# Either way it refuses to touch a checkout with uncommitted tracked changes.
set -euo pipefail
[ $# -eq 6 ] || { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
ROOT=$1 BRANCH=$2 SOURCE=$3 MODE=$4 REPO=$5 REBUILD=$6
die() { echo "pod_setup: error: $*" >&2; exit 1; }
say() { echo "pod_setup: $*"; }

cd "$ROOT"
PY=$ROOT/venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3)
old=$(git rev-parse HEAD)

# git@github.com:o/r(.git) and https://github.com/o/r(.git) are the same repo
norm() { local u=${1%.git}; u=${u#git@github.com:}; u=${u#https://github.com/}; echo "$u"; }
if [ -n "$REPO" ]; then
  cur_url=$(git remote get-url origin 2>/dev/null || true)
  if [ "$(norm "$cur_url")" != "$(norm "$REPO")" ]; then
    if [ -n "$cur_url" ] && ! git remote get-url upstream >/dev/null 2>&1; then
      git remote rename origin upstream
      git remote add origin "$REPO"
      say "origin -> $REPO (was $cur_url, now 'upstream')"
    elif [ -n "$cur_url" ]; then
      git remote set-url origin "$REPO"; say "origin -> $REPO (was $cur_url)"
    else
      git remote add origin "$REPO"; say "origin -> $REPO"
    fi
  fi
fi

case $SOURCE in
  origin) git fetch -q origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" \
            || die "cannot fetch $BRANCH from origin ($(git remote get-url origin))"
          ref=refs/remotes/origin/$BRANCH ;;
  dev)    ref=refs/remotes/dev/$BRANCH
          git rev-parse -q --verify "$ref" >/dev/null || die "$ref missing (push it first)" ;;
  *)      die "bad SOURCE $SOURCE" ;;
esac

[ -z "$(git status --porcelain --untracked-files=no)" ] \
  || die "$ROOT has uncommitted changes; commit/stash them or use --no-git"
cur=$(git symbolic-ref --short -q HEAD || true)
case $MODE in
  reset)
    git checkout -q -B "$BRANCH" "$ref" ;;
  ff)
    if [ "$cur" != "$BRANCH" ]; then
      [ "$cur" = main ] || [ "$cur" = master ] \
        || die "$ROOT is on '${cur:-detached HEAD}', not $BRANCH or main; switch it yourself or use --no-git"
      if git rev-parse -q --verify "refs/heads/$BRANCH" >/dev/null; then
        git checkout -q "$BRANCH"
      else
        git checkout -q -b "$BRANCH" "$ref"
      fi
    fi
    git merge -q --ff-only "$ref" || die "$BRANCH has diverged from $ref; not touching it" ;;
  *) die "bad MODE $MODE" ;;
esac
new=$(git rev-parse HEAD)
if [ "$old" = "$new" ]; then say "$BRANCH at $(git log --oneline -1)"
else say "$BRANCH: ${old:0:9} -> $(git log --oneline -1)"; fi

[ "$REBUILD" = never ] && exit 0

if [ "$old" != "$new" ] && ! git diff --quiet "$old" "$new" -- requirements.txt; then
  say "requirements.txt changed; pip install"
  "$PY" -m pip install -q -r requirements.txt || die "pip install failed"
fi

# ui/dist is git-ignored build output; remember which commit it was built from.
# The image ships a build of its own HEAD, i.e. $old the first time.
stamp=ui/dist/.aitk_built_rev
built=$(cat "$stamp" 2>/dev/null || echo "$old")
ui_up() { (cd scripts && "$PY" -c 'import aitk_ui; aitk_ui.get_settings()') >/dev/null 2>&1; }
wait_up() { for _ in $(seq "$1"); do sleep 2; ui_up && return 0; done; return 1; }
if [ "$REBUILD" = auto ] && git diff --quiet "$built" "$new" -- ui; then
  ui_up && exit 0
  say "UI build is current but the UI is not answering; (re)starting it"
else
  active=$(cd scripts && "$PY" -c 'import aitk_ui; print(" ".join(j["name"] for j in aitk_ui.list_jobs() if j["status"] in ("running","queued","stopping")))') \
    || die "cannot query the UI for active jobs"
  [ -z "$active" ] || die "UI code changed but job(s) $active are active; not restarting the UI under them"

  say "ui/ changed since its build (${built:0:9}); rebuilding (log: /tmp/aitk_ui_build.log)"
  (
    cd ui
    git diff --quiet "$built" "$new" -- package.json package-lock.json || npm run install_deps
    git diff --quiet "$built" "$new" -- prisma || npm run update_db
    npm run build
  ) >/tmp/aitk_ui_build.log 2>&1 || { tail -n 30 /tmp/aitk_ui_build.log >&2; die "UI build failed"; }
  echo "$new" >"$stamp"
fi

# Restart. In the docker image the `concurrently` supervisor is (via npm and
# /start.sh) the container's main process: killing it restarts the container,
# which keeps its disk (incl. this checkout and the new build) and comes back
# with the UI on the new build and the container env (HF_TOKEN,
# AI_TOOLKIT_AUTH) -- cleaner than respawning pieces of it from an ssh
# session. SIGKILL: on TERM concurrently just respawns its children.
# Exit 75 tells aitk_remote.py to reconnect and wait for the UI.
sup=$(pgrep -f '^node .*/node_modules/\.bin/concurrently .*dist/cron/worker\.js' | head -n1 || true)
if [ -n "$sup" ]; then
  say "restarting the container for the new UI build (supervisor pid $sup)"
  setsid -f bash -c "sleep 1; kill -KILL $sup" >/dev/null 2>&1 </dev/null
  exit 75
fi
ui_up && say "warning: UI rebuilt but not restarted (no image supervisor found); restart it yourself" && exit 0
die "UI rebuilt but not running and no image supervisor to restart it; start it yourself"
