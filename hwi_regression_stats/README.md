# Hand-wing index regression statistics

How well do CELLECTION's hand-wing index predictions track the observed values, on the species it was trained on and on held-out ones? This folder adds the statistics behind Figure S18: Pearson and Spearman correlations with p-values and bootstrap confidence intervals, a baseline that predicts the training mean, and observed-vs-predicted scatter plots. Reviewer 2 raised this in question 5.

## How to run

```bash
export RESULTS_DIR=/path/to/results
cd experiments-public/hwi_regression_stats
python correlation_metrics.py
```

The script reads `$RESULTS_DIR/R2-q5/Data/all_predictions_Hand-Wing.Index.csv`, one row per species (index = species name) with `original_value`, `prediction` and `split` (`train` / `test`). That file comes from a CELLECTION hand-wing-index regression run, which is not part of this folder. It needs numpy, pandas, scipy and matplotlib, and runs on a CPU in seconds.

## Files

- `correlation_metrics.py`: for the train and test splits, and for a train-mean baseline on the test split, computes Pearson and Spearman correlations with p-values and 95% bootstrap confidence intervals (5,000 resamples, seed 0), and draws observed-vs-predicted scatter plots.

## Outputs

Under `$RESULTS_DIR/R2-q5/Results/`:
- `correlation_metrics.csv`: one row each for `train`, `test` and `test_mean_baseline`, with n, then Pearson r, p and CI bounds and the same for Spearman. The baseline's prediction is constant, so its correlation columns are NaN.
- `scatter_train.pdf`, `scatter_test.pdf`: observed vs predicted hand-wing index.
