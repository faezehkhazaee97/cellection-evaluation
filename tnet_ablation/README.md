# T-Net ablation

CELLECTION's backbone has two T-Nets: TNet1 applies a learned transform to each cell's input features and TNet2 does the same for the intermediate features. Switching them on and off across five classification tasks shows how much each one contributes. The runs with all four on/off combinations make up Figure S2 (Reviewer 1, question 2).

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/tnet_ablation

# All four T-Net combinations x 3 seeds x 5 CV folds per task (Figure S2)
python four_configs/covid.py
python four_configs/cardio.py
python four_configs/cardio_biosample.py
python four_configs/lupus_disease.py
python four_configs/lupus_race.py
python plot_tnet_bars.py                                    # Figure S2 bar charts
python four_configs/plot_single_metric.py --metric AUROC    # one-metric bar charts (optional)

# Both T-Nets on vs both off x 5 seeds x 5 CV folds
python both_on/covid.py
python both_off/covid.py
# ...and likewise cardio.py, cardio_biosample.py, lupus_disease.py, lupus_race.py in each folder
```

The training scripts read `$DATA_DIR/covid/covid_Ziegler_processed.h5ad` with `covid_5folds.csv`, `$DATA_DIR/cardio/cardio_processed.h5ad` with `cardio_5folds.csv` and `cardio_biosample_id_5folds.csv`, and `$DATA_DIR/lupus/lupus_processed.h5ad` with `lupus_5folds.csv` and `race_5folds.csv`. Every fold file has the columns `samples` and `fold`.

They need the `cellection` package in a build that accepts `use_tnet1` / `use_tnet2`, plus scanpy, torch, scikit-learn, matplotlib and seaborn, and they should run on a GPU (the CPU fallback is very slow). Rough times on one GPU:
- each `four_configs/` script: COVID about 45 min, Cardio about 3.5 h, Cardio by biosample about 4.5 h, lupus ancestry about 5.5 h, lupus disease about 11 h;
- each `both_on/` or `both_off/` script: COVID about 15 min, Cardio about 2 h, lupus about 4 h;
- the plot scripts: seconds.

## Files

- `plot_tnet_bars.py`: draws the Figure S2 chart for each task from the `four_configs/` results, three panels (AUROC, macro-F1, macro AUPRC) with one bar per T-Net combination, mean ± SD across seeds.
- `four_configs/covid.py`: Ziegler COVID-19, case vs control, with both T-Nets off, both on, only TNet1 and only TNet2, 3 seeds × 5 folds; saves per-fold metrics, predictions, curves and checkpoints.
- `four_configs/cardio.py`: the same four combinations on Cardio (DCM / HCM / NF), one bag per donor.
- `four_configs/cardio_biosample.py`: Cardio again, one bag per biosample.
- `four_configs/lupus_disease.py`: lupus, SLE vs healthy.
- `four_configs/lupus_race.py`: lupus ancestry, Asian vs White (`pop_cov`).
- `four_configs/plot_single_metric.py`: one bar chart per task for whichever metric `--metric` names, from the `four_configs/` results.
- `both_on/covid.py`: Ziegler COVID-19 with both T-Nets on, 5 seeds × 5 folds; saves per-fold metrics, predictions, curves and checkpoints.
- `both_on/cardio.py`: both on, Cardio by donor.
- `both_on/cardio_biosample.py`: both on, Cardio by biosample.
- `both_on/lupus_disease.py`: both on, lupus SLE vs healthy.
- `both_on/lupus_race.py`: both on, lupus Asian vs White.
- `both_off/covid.py`: the matching run with both T-Nets off, Ziegler COVID-19, 5 seeds × 5 folds.
- `both_off/cardio.py`: both off, Cardio by donor.
- `both_off/cardio_biosample.py`: both off, Cardio by biosample.
- `both_off/lupus_disease.py`: both off, lupus SLE vs healthy.
- `both_off/lupus_race.py`: both off, lupus Asian vs White.

## Outputs

Everything goes under `$RESULTS_DIR/R1-q2/Results/`.
- `TnetConfigs/<Task>/<config>/`, where `<config>` is `tnet_both_off`, `tnet_both_on`, `tnet1_on_tnet2_off` or `tnet1_off_tnet2_on`:
  - `metrics_per_fold.csv`: one row per seed × fold with AUROC, AUPRC, macro AUPRC, macro-F1 and per-class F1 (Cardio: macro one-vs-rest AUROC, macro AUPRC and macro-F1 only);
  - `MacroAUPRC_values.csv`, `per_donor_predictions.csv` (true label and predicted probabilities for each held-out sample), and checkpoints in `seed_<s>/fold_<k>/`;
  - for the binary tasks also `conf_mats.npy`, `confusion_matrix_mean.pdf`, `mean_ROC_curve.pdf` and `mean_PR_curve.pdf`.
- `TnetConfigs/plots/<Task>_tnet_metrics.pdf`: the Figure S2 panels. `TnetConfigs/plots/<Task>_<metric>_tnet_bars.pdf`: the one-metric charts.
- `WithTnet/<Task>/` and `WithoutTnet/<Task>/`: `metrics_per_fold.csv`, `MacroAUPRC_values.csv`, `per_donor_predictions.csv` and checkpoints, plus, for the binary tasks, `conf_mats.npy` and PNGs of the mean ROC curve, mean PR curve and mean confusion matrix.
