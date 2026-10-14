# MMIL baseline

How far does a plain cell-level classifier get on the three binary benchmark tasks (Ziegler COVID-19, lupus SLE vs healthy, lupus Asian vs White)? This MMIL-style baseline gives every cell its donor's label, fits a ridge logistic regression (glmnet, in R) on the training donors' cells for each donor fold, and scores the cells of the held-out donors. It was added for Reviewer 1, question 5.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/mmil_baseline
R="$RESULTS_DIR/R1-q5/Results"; mkdir -p "$R"

python prep_covid.py
Rscript mmil_covid.R "$R/cells.csv" "$R/folds.csv" "$R"

python prep_lupus_disease.py
python make_stratified_folds.py   # stratified donor folds -> Lupus_Disease/folds_stratified.csv
Rscript mmil_lupus_disease.R "$R/Lupus_Disease/cells.csv" "$R/Lupus_Disease/folds_stratified.csv" "$R/Lupus_Disease"

python prep_lupus_race.py
Rscript mmil_lupus_race.R "$R/Lupus_Race/cells.csv" "$R/Lupus_Race/folds.csv" "$R/Lupus_Race"
```

The R scripts accept any donor-to-fold table. `folds.csv` is the benchmark's predefined split; `folds_stratified.csv` is a stratified 5-fold donor split that `make_stratified_folds.py` builds for `Lupus_Disease` (point its two paths at `Lupus_Race/` to do the same there). To run every task on the predefined folds in one command, use `python runner/run_mmil.py --dataset all` or `bash scripts/run_all_mmil_benchmarks.sh` (see `runner/README.md`).

What it needs:
- Data: `$DATA_DIR/covid/covid_Ziegler_processed.h5ad` and `covid_5folds.csv`; `$DATA_DIR/lupus/lupus_processed.h5ad`, `lupus_5folds.csv` and `race_5folds.csv`.
- Python with scanpy/anndata, pandas, numpy and scikit-learn; R with Matrix, glmnet, pROC, dplyr and caret, with `Rscript` on the PATH.
- CPU only, but plenty of memory: each lupus `cells.csv` stores every gene densely (about 1 GB) and R reads it whole, so use a compute node.
- About 25-40 min per R model run (5 folds); the prep scripts are quicker.

## Files

- `prep_covid.py`: writes the COVID per-cell table `cells.csv` (cell ID, donor, every gene, the donor's binary label; all cells) and the donor-to-fold table `folds.csv`.
- `prep_lupus_disease.py`: writes the SLE-vs-healthy `cells.csv` (10% of each donor's cells, seed 0) and `folds.csv` to `Lupus_Disease/`.
- `prep_lupus_race.py`: the Asian-vs-White `cells.csv` (same 10% subsample) and `folds.csv`, in `Lupus_Race/`.
- `make_stratified_folds.py`: stratified 5-fold donor split from `Lupus_Disease/cells.csv`, saved as `Lupus_Disease/folds_stratified.csv`.
- `mmil_covid.R`: per fold, fits `cv.glmnet` (alpha 0, lambda.min) on the training donors' cells, predicts the held-out donors' cells, and writes the predictions and AUROC / F1 / macro AUPRC.
- `mmil_lupus_disease.R`: the same model for SLE vs healthy.
- `mmil_lupus_race.R`: the same model for Asian vs White.
- `runner/run_mmil.py`: one command per task (or `--dataset all`); builds `cells.csv` / `folds.csv` from `$DATA_DIR` with the predefined folds and calls the matching R script.
- `runner/README.md`: options for `run_mmil.py`.
- `scripts/run_all_mmil_benchmarks.sh`: wrapper for `run_mmil.py --dataset all`.

## Outputs

Everything goes under `$RESULTS_DIR/R1-q5/Results/`. The prep scripts put COVID at the top level and the lupus tasks in `Lupus_Disease/` and `Lupus_Race/`; `run_mmil.py` puts COVID in `Covid/`.
- `cells.csv`, `folds.csv`: the model inputs. `label_counts.csv`: cells per label (lupus prep scripts and `run_mmil.py`).
- `folds_stratified.csv`: stratified donor folds (`make_stratified_folds.py`).
- `fold_<k>_preds.csv`: one row per held-out cell with cell ID, donor, label and predicted probability.
- `MMIL_summary.csv`: AUROC, F1 and macro AUPRC for each fold.
