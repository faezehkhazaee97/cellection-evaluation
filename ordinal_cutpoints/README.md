# Learned vs fixed ordinal cut points

For ordinal outcomes CELLECTION gives each patient one score and turns it into a class with K-1 cut points. Here we ask whether those cut points should be learned or fixed, predicting COVID-19 severity on an 8-level scale in the Haniffa data: learned cut points from two starting ranges against fixed cut points in three settings (Figure S10). Reviewer 1 raised this in question 11.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/ordinal_cutpoints
python learned_cutpoints.py    # learned, starting from [-5, 5] and from [-10, 10]
python fixed_cutpoints.py      # fixed: package default, [-5, 5], [-10, 10]
python plot_spearman.py        # run in a folder that holds its two input CSVs (see Files)
```

Both training scripts read `$DATA_DIR/haniffa21_intersection.h5ad` with the obs columns `Worst_Clinical_Status` and `sample_id`. They need the `cellection` package with the ordinal task (`train_cutpoints`), plus scanpy, torch, scipy, scikit-learn and seaborn, and use a GPU when one is available. Each takes about 1 h on one GPU (7 seeds for every setting).

## Files

- `learned_cutpoints.py`: for seeds 7, 8, 9, 10, 6, 5 and 2, trains ordinal CELLECTION with cut points learned from two starting ranges. On the validation patients it reports Spearman and Pearson correlations with the true severity, both from the raw ordinal score and from the decoded class, plus the accuracy of the decoded class, and draws scatter plots.
- `fixed_cutpoints.py`: the same with the cut points frozen, at the package default, [-5, 5] or [-10, 10]. The package default is already evenly spaced over [-10, 10], so the first and third settings start from the same values. With `TRAIN_CUTPOINTS = True` it instead runs one learned setting into `with_new/trained/`.
- `plot_spearman.py`: box-and-strip plot of Spearman correlation, learned vs fixed, from `haniffa_cellection_metrics_cutpoints_True.csv` and `haniffa_cellection_metrics_cutpoints_False.csv` in the working folder (columns `train_cutpoints` and `spearman`). The training scripts write one file per setting and call that column `spearman_logits`, so the two inputs have to be assembled from their output first.

## Outputs

- `learned_cutpoints.py` writes to `$RESULTS_DIR/R1-Q11/Results/with_new/`:
  - `haniffa_cellection_metrics_cutpoints_True_{trained_init_-5_5,trained_init_-10_10,all_experiments}.csv`: one row per seed (and per setting in `all_experiments`), with Spearman and Pearson and their p-values from the ordinal score (`*_logits`) and from the decoded class (`*_decoded`), and `accuracy_decoded`;
  - `haniffa_cellection_seed<S>_cutpoints_True_<exp>_{true,pred}_labels.csv`: true and predicted severity per validation patient;
  - `haniffa_severity_scatter_seed<S>_cutpoints_True_<exp>.pdf`;
  - `checkpoints_<exp>/severity_rank_ordinal_SEED<S>/`: model checkpoints.
- `fixed_cutpoints.py` writes to `$RESULTS_DIR/R1-Q11/Results/without_new/{default,fixed_-5_5,fixed_-10_10}/`: the per-setting metrics CSV, the label CSVs and the scatter plots, with `cutpoints_False_<exp>` in the names, and checkpoints in `severity_rank_ordinal_SEED<S>/`. It writes no CSV pooling the settings.
- `plot_spearman.py` writes `cutpoint_ablation_spearman.pdf` to the working folder.
