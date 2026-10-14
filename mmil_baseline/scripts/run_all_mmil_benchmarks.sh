#!/usr/bin/env bash
# MMIL baseline on covid, lupus_disease and lupus_race in one go (runner/run_mmil.py --dataset all).
# Export DATA_DIR and RESULTS_DIR and put Rscript on the PATH first; PYTHON_BIN picks the Python
# interpreter (default python3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNNER_DIR="$CODE_DIR/runner"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[RUN] MMIL all datasets"
"$PYTHON_BIN" "$RUNNER_DIR/run_mmil.py" --dataset all

echo "[DONE] MMIL benchmarks finished."
