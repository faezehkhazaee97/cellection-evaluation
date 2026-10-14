# Cell-count imbalance across samples

Samples contribute very different numbers of cells, so does CELLECTION simply do better on the cell-rich ones? For each task the samples are split into quantile bins of cell count, fixed on the full data, and held-out AUROC and F1 are computed per bin under 5-fold CV with 10 seeds. The largest bins are then cut down to the size of the middle ones, the model is retrained, and the same bins are scored again (Figure S4). This was Reviewer 1's question 8.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/cell_count_imbalance

# covid, cardio, cardio_biosample, lupus_disease: baseline -> downsample and retrain -> plots, in this order
python covid/before_downsampling.py
python covid/downsample_and_retrain.py
python covid/plot_before_after.py
# ...the same three commands for cardio/, cardio_biosample/ and lupus_disease/

# lupus_race: one script runs the baseline and the downsampled retraining
python lupus_race/before_and_after_downsampling.py
python lupus_race/plot_before_after.py

python export_grouped_bin_excel.py     # all plotted values in one workbook
```

The scripts read `$DATA_DIR/covid/covid_Ziegler_processed.h5ad` and `covid_5folds.csv`; `$DATA_DIR/cardio/cardio_processed.h5ad`, `cardio_5folds.csv` and `cardio_biosample_id_5folds.csv`; and `$DATA_DIR/lupus/lupus_processed.h5ad`, `lupus_5folds.csv` and `race_5folds.csv`. Each fold file has the label as its first (index) column, then `samples` and `fold`.

They need the `cellection` package, scanpy, torch, scikit-learn, seaborn and matplotlib, plus pandas and openpyxl for the Excel export. Use a GPU if you can: every baseline or retraining pass fits 50 models (5 folds x 10 seeds), although the scripts fall back to CPU. On one GPU a training script takes about 10 min for COVID and about 1 h for Cardio, Cardio by biosample and lupus. The plot scripts take seconds.

## Files

- `covid/before_downsampling.py`: splits Ziegler COVID-19 donors into 8 quantile bins of cells per donor, trains CELLECTION over the 5 folds for seeds 0-9, and writes per-bin AUROC and F1 with the bin edges and the donor-to-bin map.
- `covid/downsample_and_retrain.py`: cuts donors in bins 7-8 down to the mean of the bin-5 and bin-6 median cell counts, retrains with the same folds and seeds, and scores each donor under its original bin.
- `covid/plot_before_after.py`: grouped bars of per-bin AUROC and MacroF1, before vs after, for COVID.
- `cardio/before_downsampling.py`: the 4-bin baseline for Cardio (DCM / HCM / NF), one bag per donor.
- `cardio/downsample_and_retrain.py`: caps bin-4 donors at the mean of the bin-2 and bin-3 medians, then retrains and re-scores per bin.
- `cardio/plot_before_after.py`: before/after bars for Cardio.
- `cardio_biosample/before_downsampling.py`: the Cardio baseline with one bag per biosample.
- `cardio_biosample/downsample_and_retrain.py`: downsampling and retraining for Cardio by biosample.
- `cardio_biosample/plot_before_after.py`: before/after bars for Cardio by biosample.
- `lupus_disease/before_downsampling.py`: the 4-bin baseline for lupus, SLE vs healthy.
- `lupus_disease/downsample_and_retrain.py`: downsampling and retraining for SLE vs healthy.
- `lupus_disease/plot_before_after.py`: before/after bars for SLE vs healthy.
- `lupus_race/before_and_after_downsampling.py`: lupus Asian vs White, with the 4-bin baseline and the downsampled retraining in one script.
- `lupus_race/plot_before_after.py`: before/after bars for Asian vs White.
- `export_grouped_bin_excel.py`: gathers the plotted per-bin values and the per-seed/fold results of all five tasks into one Excel workbook.

## Outputs

Everything goes under `$RESULTS_DIR/R1-q8/Results/`.
- `covid_Ziegler/CELECTION/bin_8/`: `global_bin_edges.npy`, `donor_bin_map.csv` (bin per donor), `cellection_results_summary.csv` (metrics per seed and fold), `cellection_global_bins.csv` (AUROC and F1 per bin), `global_auroc_vs_cell_bin.pdf`, `global_macrof1_vs_cell_bin.pdf`.
- `covid_Ziegler/CELECTION/bin_8_subsample_retrain/`: `donor_bin_map_original.csv`, `cellection_results_summary.csv`, `cellection_global_bins_retrain_downsampled.csv`, `global_auroc_vs_cell_bin_retrain.pdf`, `global_macrof1_vs_cell_bin_retrain.pdf`.
- `covid_Ziegler/CELECTION/auroc_grouped_bins_6.pdf` and `macrof1_grouped_bins_6.pdf`: the COVID before/after plots (all 8 bins, despite the name).
- `<task>/CELECTION/` for cardio, cardio_biosample, lupus_disease and lupus_race:
  - `4-bin/`: `global_bin_edges.npy`, `donor_bin_map.csv`, `results_summary.csv` (per seed and fold), `per_bin_metrics.csv`, `AUROC_per_bin.pdf`, `F1_per_bin.pdf`;
  - `4-bin_downsample/`: `summary_downsampled.csv`, `per_bin_metrics.csv`, `AUROC_per_bin.pdf`, `MacroF1_per_bin.pdf`;
  - `4-bin_plots/`: `<prefix>_4bin_auroc_before_after.pdf` and `<prefix>_4bin_macrof1_before_after.pdf`, where the prefix is `cardio`, `cardiobiosample`, `lupusdisease` or `lupusrace`.
- `grouped_bin_plot_values.xlsx`: one sheet per task.
- Model checkpoints: `./checkpoints_<task>_<before|downsampled>/` in the folder the scripts are run from.

F1 definitions differ by task. The COVID MacroF1 column is the binary F1 of label column 1. In the lupus per-bin tables it is the binary F1 of "sle" or "WHITE", except in lupus_disease after downsampling, where it is the macro F1. Cardio uses the macro F1 throughout. Each script's docstring gives its definition.
