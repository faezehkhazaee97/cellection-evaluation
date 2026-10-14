# sciLaMA embeddings vs raw expression

Can CELLECTION work from precomputed sciMultiLaMA cell embeddings instead of raw gene expression, and how much training time does that save? Both inputs are tested on donor-level Alzheimer's disease vs control prediction in SEA-AD MTG glia, and a one-epoch probe times each arm. Raised by Reviewer 1, question 3.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/seaad_scilama_vs_raw
python train_raw_expression.py
python train_scilama.py
```

The training-time comparison is a Slurm GPU array job with one task per arm:

```bash
export CODE_DIR=$PWD VENV_DIR=/path/to/.venv
mkdir -p "$RESULTS_DIR/R1-q3/Results/timing_probe/logs" && cd "$RESULTS_DIR/R1-q3"
sbatch "$CODE_DIR/timing/run_timing_probe.sbatch"      # set --account and --partition first
python "$CODE_DIR/timing/summarize_timing_probe.py"     # after both array tasks have finished
```

What it needs:
- `$DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad` (`obs` with `donor_id` and `disease`), and for the sciLaMA arm `$DATA_DIR/sea-ad/MTG_sciMultiLaMA_CELL_embedding.csv` (one row per cell, indexed by the cell IDs in `adata.obs_names`). The embeddings come from sciMultiLaMA, which is not included.
- The `cellection` package, scanpy, torch, scikit-learn and pandas.
- A GPU; the CPU fallback is slow.
- On one GPU the raw-expression arm runs at about 8 s per epoch. With at most 200 epochs (usually fewer, thanks to early stopping) that is up to about 27 min per seed and 3.5 h for all 8 seeds. The sciLaMA arm runs at about 1.4 s per epoch, under 5 min per seed. Each timing task takes a few minutes.

## Files

- `train_raw_expression.py`: for seeds 17, 16, 9 and 18-22, splits donors 80/20 (stratified, cached in `seaad_splits/`), trains CELLECTION on normalized expression and scores the test donors; writes per-seed metrics and predictions.
- `train_scilama.py`: the same protocol and donor splits, with each cell given as its sciMultiLaMA embedding; scores the test donors with the best checkpoints.
- `timing/timing_raw_expression_seed18_1ep.py`: the raw-expression arm cut down to seed 18 and one epoch, with a timer around training; prints a `TIMING_ONLY … seconds_per_epoch=…` line and exits.
- `timing/timing_scilama_seed18_1ep.py`: the same one-epoch timing run for the sciLaMA arm.
- `timing/run_timing_probe.sbatch`: Slurm array job; task 0 runs the raw-expression timing script, task 1 the sciLaMA one, and logs go to `Results/timing_probe/logs/` under the submit folder.
- `timing/summarize_timing_probe.py`: reads the two logs, scales each arm's seconds per epoch to 200 epochs, and prints the estimated training times, time saved, percent reduction and speed-up.

## Outputs

All under `$RESULTS_DIR/R1-q3/Results/`.
- `measurement_baseline_NoFolds/` (raw expression) and `sciMultiLaMA_NoFolds/` (sciLaMA), each with:
  - `metrics_5seed.csv`: one row per seed (all 8, despite the name) with AUROC, AUPRC, macro AUPRC, macro F1, per-class F1 and the confusion matrix;
  - `predictions_5seed.csv`: seed, donor, true label and predicted score for each test donor;
  - `seed_<SEED>/`: model checkpoints.
- `seaad_splits/disease_seed_<SEED>_split.csv`: the donor train/test split both arms use.
- `timing_probe/logs/celect_timing_<job>_<task>.{out,err}`: Slurm logs with the per-epoch timing lines; `timing_probe/<arm>_seed18_1ep/`: working folders of the timing runs.
