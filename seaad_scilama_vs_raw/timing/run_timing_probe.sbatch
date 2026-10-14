#!/bin/bash
#SBATCH --job-name=celect_timing
#SBATCH --partition=YOUR_PARTITION            # set your own GPU partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%1
#SBATCH --output=Results/timing_probe/logs/%x_%A_%a.out
#SBATCH --error=Results/timing_probe/logs/%x_%A_%a.err

# One timed training epoch per arm on seed 18: array task 0 is raw expression, task 1 sciLaMA.
# summarize_timing_probe.py scales the per-epoch times to 200 epochs.
#
# Submit from $RESULTS_DIR/R1-q3. The log paths above are relative to the submit directory, and
# summarize_timing_probe.py looks for the logs there.
#   export DATA_DIR=... RESULTS_DIR=... CODE_DIR=/path/to/experiments-public/seaad_scilama_vs_raw VENV_DIR=/path/to/.venv
#   mkdir -p "$RESULTS_DIR/R1-q3/Results/timing_probe/logs" && cd "$RESULTS_DIR/R1-q3"
#   sbatch "$CODE_DIR/timing/run_timing_probe.sbatch"
#   python "$CODE_DIR/timing/summarize_timing_probe.py"    # after both array tasks finish
#
# Only timing_raw_expression_seed18_1ep.py and timing_scilama_seed18_1ep.py exist in timing/, so
# leave TIMING_SEED=18 and TIMING_EPOCHS=1.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
source "${VENV_DIR:-.venv}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TIMING_SEED="${TIMING_SEED:-18}"
TIMING_EPOCHS="${TIMING_EPOCHS:-1}"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    LABEL="raw_expression"
else
    LABEL="scilama"
fi

TIMING_SCRIPT="${CODE_DIR:?set CODE_DIR to experiments-public/seaad_scilama_vs_raw}/timing/timing_${LABEL}_seed${TIMING_SEED}_${TIMING_EPOCHS}ep.py"

/usr/bin/time -f "SLURM_WALL_SECONDS=%e" python "${TIMING_SCRIPT}"
