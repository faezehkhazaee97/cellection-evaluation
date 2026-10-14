# Attention ablation

If CELLECTION's attention picks out the cells a prediction depends on, taking the highest-attention cells out of a held-out sample should hurt the frozen model more than taking out the same number of random cells. These scripts run that comparison on five tasks, removing 5 to 50% of each test sample's cells (Figure S7). The experiment answers Reviewer 2's question 4.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/attention_ablation
python covid.py
python cardio.py
python cardio_biosample.py
python lupus_disease.py
python lupus_race.py
python plot_ablation_auprc.py --csv "$RESULTS_DIR/R2-Q4/Results/Covid/ablation_results.csv" --save-dir DIR   # optional re-plot
python export_ablation_plot_excel.py      # after all five tasks have run
```

Each training script fits 50 models (10 seeds x 5 folds) and wants a GPU; the CPU fallback is slow. The runs were not timed. The training part is the same size as the `cell_count_imbalance` baselines (about 10 min for COVID up to about 1 h for Cardio and lupus on one GPU), and the ablation passes come on top of that.

Inputs, relative to `$DATA_DIR` (fold files have the columns `samples` and `fold`):
- COVID: `covid/covid_Ziegler_processed.h5ad` and `covid/covid_5folds.csv`.
- Cardio: `cardio/cardio_processed.h5ad`, with `cardio/cardio_5folds.csv` for `cardio.py` or `cardio/cardio_biosample_id_5folds.csv` for `cardio_biosample.py`.
- Lupus: `lupus/lupus_processed.h5ad`, with `lupus/lupus_5folds.csv` for `lupus_disease.py` or `lupus/race_5folds.csv` for `lupus_race.py`.

Python packages: `cellection`, scanpy, torch, scipy, scikit-learn, matplotlib and seaborn, plus openpyxl for the Excel export.

## Files

- `covid.py`: Ziegler COVID-19, case vs control, one bag per donor. Trains seeds 0-9 x 5 folds, then removes 5, 10, 20, 25 or 50% of each test donor's cells (highest attention, or random), re-scores with the frozen model, writes per-run metrics, pivots, curves and top-minus-random bar plots, and prints Wilcoxon tests.
- `cardio.py`: the same on Cardio (DCM / HCM / NF), one bag per donor, with macro metrics.
- `cardio_biosample.py`: Cardio again, with one bag per biosample.
- `lupus_disease.py`: lupus, SLE vs healthy.
- `lupus_race.py`: lupus, Asian vs White (`pop_cov`).
- `plot_ablation_auprc.py`: redraws the AUPRC curve and delta-AUPRC bar plot (PDF and PNG) and the AUPRC pivot from an existing `ablation_results.csv`.
- `export_ablation_plot_excel.py`: gathers the values behind every ablation plot of the five tasks into one Excel workbook.

## Outputs

Each task writes to `$RESULTS_DIR/R2-Q4/Results/<Task>/`, with `<Task>` one of `Covid`, `Cardio`, `Cardio_Biosample`, `Lupus_Disease`, `Lupus_Race`:
- `ablation_results.csv`: one row per seed x fold x removed fraction x setting (`top` / `random`), with AUROC, AUPRC and F1 after removal and the unablated `baseline_AUROC` and `baseline_AUPRC`.
- `ablation_pivot_{AUROC,AUPRC,F1}.csv`: top and random side by side per seed x fold x fraction, with their difference.
- `ablation_{auroc,auprc,f1}_curve.pdf` and `ablation_delta_{auroc,auprc,f1}_barplot.pdf`.
- `Covid`, `Cardio` and `Cardio_Biosample` also get `MacroAUPRC_values.csv` and `per_donor_predictions.csv`; `Lupus_Disease` and `Lupus_Race` get `AUPRC_values.csv` and `metrics_per_fold.csv` instead.

`plot_ablation_auprc.py` writes `ablation_auprc_curve.{pdf,png}`, `ablation_delta_auprc_barplot.{pdf,png}` and `ablation_pivot_AUPRC.csv` to `--save-dir` (by default the CSV's folder). `export_ablation_plot_excel.py` writes `$RESULTS_DIR/R2-Q4/Results/ablation_plot_values.xlsx`.

Model checkpoints go under the folder the scripts are run from: `./checkpoints_cardio/`, `./checkpoints_cardio_biosample/`, and `./<task_key>_classification_SEED<seed>/` for the other tasks.
