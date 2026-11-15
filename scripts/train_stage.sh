#!/usr/bin/env bash
# Run one or more training stages of a job through the AI Toolkit UI and wait.
#
#     train_stage.sh [-g GPUS] [--stall MIN] <job_name> <config.yaml>...
#     train_stage.sh <job_name> <config.yaml> [gpus]          (legacy form)
#
# Thin wrapper: picks the toolkit's python (venv/ if present, else python3, as
# in the docker image) and runs train_stage.py next to it. See --help.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)
ROOT=${AITK_ROOT:-$(dirname -- "$HERE")}
PY=$ROOT/venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3 || command -v python)
exec "$PY" "$ROOT/scripts/train_stage.py" "$@"
