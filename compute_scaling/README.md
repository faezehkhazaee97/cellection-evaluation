# Computational cost and scaling

How does CELLECTION's training cost grow with the number of patients and with the number of cells per patient? On the Ziegler COVID-19 data, training time and peak GPU memory are measured over a grid of 4 patient counts and 4 bag sizes (Figure S6). This was Reviewer 1's question 13.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/compute_scaling
python covid_scaling.py
python export_nonbar_plot_excel.py      # Excel workbook of the line- and scatter-plot values
```

The script reads `$DATA_DIR/covid/covid_Ziegler_processed.h5ad` and `$DATA_DIR/covid/covid_5folds.csv`. It needs the `cellection` package, anndata, scanpy, torch, scikit-learn, seaborn and matplotlib (pandas, numpy and openpyxl for the export), and a CUDA GPU, since GPU memory is one of the measurements. All 160 training runs take about 1 h on one GPU; the timings depend on the GPU model.

## Files

- `covid_scaling.py`: for seeds 0-1 and folds 1-5, trains CELLECTION on every combination of 14, 29, 58 and 116 training patients with 119, 409, 815 and 2,140 cells per patient (patients drawn with replacement and duplicated when there are too few; bags subsampled or padded to the target size), records training time, time per epoch, peak GPU memory and test metrics, and draws the scaling plots.
- `export_nonbar_plot_excel.py`: writes the values behind the three line plots and the two scatter plots (with their least-squares fit) from `scaling_runtime_gpu.csv` to an Excel workbook.

## Outputs

Everything goes under `$RESULTS_DIR/R1-Q13/Code/Results/Covid/`.
- `scaling_runtime_gpu.csv`: one row per run with seed, fold, train_patients, cells_per_patient, total_cells_train, gpu_memory_mb, epochs_run, epoch_time_sec, total_time_sec and n_genes. epochs_run is always the 200-epoch cap, so epoch_time_sec is total_time_sec / 200.
- `metrics_across_scaling.csv`: AUROC, AUPRC, macro AUPRC, macro-F1, per-class F1 and the confusion matrix for each seed, fold, patient count and bag size; `per_donor_predictions.csv`: true label and predicted probability per test donor; `confusion_matrices.npy`: row-normalised confusion matrices.
- Plots: `epoch_time_vs_cells.pdf`, `total_time_vs_cells.pdf`, `gpu_memory_vs_cells.pdf`, `gpu_memory_vs_total_cells.pdf`, `total_time_vs_total_cells.pdf`, `total_cells_barplot.pdf`, `total_time_barplot.pdf`, `gpu_memory_barplot.pdf`.
- `covid_nonbar_plot_values.xlsx`: from `export_nonbar_plot_excel.py`.
- Model checkpoints in `./disease_id_classification_SEED<seed>/` in the folder the script is run from.
