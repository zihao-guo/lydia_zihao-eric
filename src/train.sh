#!/usr/bin/env bash
# One command from a fresh clone to a trained, evaluated model:
#   ./src/train.sh          full run (all candidates, grid search, ~4 min)
#   ./src/train.sh --quick    smoke run on a stratified subsample; writes to
#                         /tmp/transaction-labeling-quick (unless --out is given)
#                         so the shipped artifact in models/ is never overwritten
# Uses $PYTHON if set (default: python), installs the pinned dependencies,
# trains, then re-checks the selected artifact on the test period
# (skipped for --quick).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
"$PY" -m pip install -r requirements.txt
args=("$@")
if [[ " $* " == *" --quick "* && " $* " != *" --out"* ]]; then
  args+=(--out /tmp/transaction-labeling-quick)
fi
"$PY" -m src.train "${args[@]}"
if [[ " $* " != *" --quick "* ]]; then "$PY" -m src.evaluate; fi
