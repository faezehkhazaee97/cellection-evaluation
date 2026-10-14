#!/bin/bash
#SBATCH --job-name=attn_hijack
#SBATCH --partition=YOUR_PARTITION            # set your own GPU partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_attn_hijack_%j.out     # #SBATCH lines cannot expand $RESULTS_DIR; log goes to the submit dir

# Submit from experiments-public/batch_correction_simulation/simulations with DATA_DIR and
# RESULTS_DIR exported (VENV_DIR defaults to .venv in the submit folder):
#   export DATA_DIR=/path/to/Data RESULTS_DIR=/path/to/results VENV_DIR=/path/to/.venv
#   sbatch submit_nk_hijack_test.sbatch
# Run simulate_nk_hijack.py first (no GPU needed); it writes
# $DATA_DIR/pbmc3k/pbmc_batch_confounded_nk_hijack.h5ad, which run_nk_hijack_test.py reads.
source "${VENV_DIR:-.venv}/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python3 run_nk_hijack_test.py
