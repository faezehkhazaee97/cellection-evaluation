#!/usr/bin/env bash
# Figure 2 without retraining: rebuilds the summaries, the combined table and the bubble plots from the per-run
# *_metrics_all.csv / *_results_mean.csv files already under $RESULTS_DIR/R2-q1/Results/.
# Run from experiments-public/benchmark_baselines with RESULTS_DIR exported (the last message echoes it).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$CODE_DIR/.." && pwd)"
RUNNER_DIR="$CODE_DIR/runners"
PYTHON_BIN="${PYTHON_BIN:-python3}"

run_raw() {
  local rel="$1"
  echo "[RUN] $rel"
  "$PYTHON_BIN" "$RUNNER_DIR/run_raw_script.py" "$rel"
}

run_raw "ct_pseudobulk/summarize_linear_baselines.py"
run_raw "celection/summarize.py"
run_raw "protocell4p/summarize.py"
run_raw "deepset/summarize.py"
run_raw "cloudpred/summarize.py"

run_raw "plot/combine_all_test_summaries.py"
run_raw "plot/bubbleplot_from_combined.py"

echo "[DONE] Fig. 2 outputs regenerated under ${RESULTS_DIR}/R2-q1/Results/."
