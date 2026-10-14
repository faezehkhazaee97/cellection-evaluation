# Minimum number of training patients

How few labelled patients does CELLECTION need? We train it on near-balanced subsets of 5 to 80 Haniffa COVID-19 patients (Healthy/Non-covid vs Severe/Critical/Death) and follow how patient-level performance changes with the size of the training set. This came out of Reviewer 1, question 10.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/min_patients_covid
python min_patients.py
```

The script reads `$DATA_DIR/haniffa21_intersection.h5ad`, which must have the obs columns `Worst_Clinical_Status` and `patient_id`. It imports the `cellection` package, scanpy, torch, scikit-learn, matplotlib and seaborn, and runs on a GPU when there is one (CPU otherwise). The whole grid, at most 50 runs (5 seeds x 10 training sizes), takes about 20-30 min on one GPU.

## Files

- `min_patients.py`: binarises severity (Healthy and Non-covid = 0; Severe, Critical and Death = 1; other statuses dropped). For seeds 0-4 and training sizes 5, 10, 15, 20, 25, 30, 35, 40, 60 and 80 it draws a near-balanced patient subset, trains CELLECTION with 30% of those patients held out for validation, scores the validation patients and plots performance against the number of patients. The validation patients also drive early stopping; there is no separate test set. Sizes that need more cases or controls than the data has are skipped.

## Outputs

Everything goes to `$RESULTS_DIR/R1-q10/Results/Haniffa_min_patients/`:

- `haniffa_binary_min_patients_metrics.csv`: one row per seed and training size, with train and validation patient counts (total, case, control), AUROC, AUPRC and macro-F1.
- `haniffa_binary_min_patients_plot.pdf`: each metric averaged over seeds, against the number of patients.
- `seed_<S>_n<N>/`: model checkpoints for each run.
