#!/usr/bin/env bash
# The whole Figure 2 benchmark: all 7 model families on all 5 tasks (10 seeds x 5 folds each), then the summaries,
# the combined table and the bubble plots. Very long; the deep models use a GPU when there is one.
# Run from experiments-public/benchmark_baselines with DATA_DIR and RESULTS_DIR exported, plus BIOPOINTNET_ROOT for
# CELECTION and an importable `cloudpred` package for DeepSet and CloudPred. PYTHON_BIN picks the interpreter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$CODE_DIR/.." && pwd)"
RUNNER_DIR="$CODE_DIR/runners"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_cmd() {
  echo "[RUN] $*"
  "$@"
}

run_runner() {
  local runner="$1"
  shift
  run_cmd "$PYTHON_BIN" "$RUNNER_DIR/$runner" "$@"
}

run_raw() {
  local rel="$1"
  shift
  run_cmd "$PYTHON_BIN" "$RUNNER_DIR/run_raw_script.py" "$rel" "$@"
}

run_runner "run_celection.py" --dataset all
run_runner "run_protocell4p.py" --dataset all
run_runner "run_deepset.py" --dataset all
run_runner "run_cloudpred.py" --dataset all

# The three linear baselines share one summary script, so it runs once, after all three.
run_runner "run_ct_pseudobulk.py" --dataset all --no-summary
run_runner "run_ct_proportion.py" --dataset all --no-summary
run_runner "run_pseudobulk.py" --dataset all --no-summary

run_raw "ct_pseudobulk/summarize_linear_baselines.py"
run_raw "plot/combine_all_test_summaries.py"
run_raw "plot/bubbleplot_from_combined.py"

echo "[DONE] Full benchmark run + summaries + Fig. 2 outputs generated."
