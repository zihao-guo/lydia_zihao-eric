#!/usr/bin/env bash
# One command from a fresh clone to a trained, evaluated model:
#   ./train.sh            full run (all candidates, grid search, ~4 min)
#   ./train.sh --quick    smoke run on a stratified subsample
# Uses $PYTHON if set (default: python), installs the pinned dependencies,
# trains, then re-checks the selected artifact on the test period.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
"$PY" -m pip install -r requirements.txt
"$PY" -m src.train "$@"
if [[ " $* " != *" --quick "* ]]; then "$PY" -m src.evaluate; fi
