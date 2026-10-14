# MMIL runner

`run_mmil.py` runs the MMIL baseline for one task or all three (`covid`, `lupus_disease`, `lupus_race`). For each task it builds `cells.csv` (one row per cell: cell ID, donor, every gene, the donor's binary label) and `folds.csv` (donor → predefined benchmark fold) from `$DATA_DIR`, then calls the matching R script (`mmil_covid.R`, `mmil_lupus_disease.R` or `mmil_lupus_race.R`) on those files.

## How to run

From `experiments-public/mmil_baseline`, with `DATA_DIR` and `RESULTS_DIR` exported and `Rscript` on the PATH:

```bash
python runner/run_mmil.py --dataset covid
python runner/run_mmil.py --dataset all
python runner/run_mmil.py --dataset lupus_race --prepare-only   # build the input tables only
python runner/run_mmil.py --dataset covid --dry-run             # print the commands only
```

`--rscript PATH` picks the R interpreter. `--sample-frac F` sets the fraction of each donor's cells to keep; by default that is 1.0 for COVID and 0.1 for the two lupus tasks, sampled with seed 0.

## Outputs

`$RESULTS_DIR/R1-q5/Results/{Covid,Lupus_Disease,Lupus_Race}/`: `cells.csv`, `folds.csv`, `label_counts.csv`, `fold_<k>_preds.csv` (per held-out cell: cell ID, donor, label, predicted probability) and `MMIL_summary.csv` (AUROC, F1 and macro AUPRC per fold).
