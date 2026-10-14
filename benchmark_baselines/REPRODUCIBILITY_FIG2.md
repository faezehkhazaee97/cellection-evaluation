# Reproducing Figure 2 and Table S3

Run from `experiments-public/benchmark_baselines` with `DATA_DIR` and `RESULTS_DIR` exported, plus `BIOPOINTNET_ROOT` and `CLOUDPRED_PATH` where needed.

1. Train and evaluate every model on every task. This is the long step and wants a GPU.
   ```bash
   bash scripts/run_all_benchmark_experiments.sh
   ```
   Each family's runner in `runners/` goes through the five tasks (10 seeds × 5 folds each); the summaries, the combined table and the plots follow.

2. Rebuild the figure from per-run results that already exist (seconds):
   ```bash
   bash scripts/reproduce_fig2_from_results.sh
   ```
   This runs the five summary scripts (`ct_pseudobulk/summarize_linear_baselines.py` and `summarize.py` in `celection/`, `protocell4p/`, `deepset/` and `cloudpred/`), then `plot/combine_all_test_summaries.py` and `plot/bubbleplot_from_combined.py`.

3. Supplementary hyper-parameter tables (the two `Code/` folders below must exist):
   ```bash
   python scripts/generate_comprehensive_benchmark_tables.py
   ```

Outputs:
- `$RESULTS_DIR/R2-q1/Results/all_models_test_summary.csv`: mean test AUROC, macro-F1 and macro AUPRC per model and task (Table S3).
- `$RESULTS_DIR/R2-q1/Results/plot/`: the Figure 2 bubble plots and mean-rank bar plots.
- `$RESULTS_DIR/R2-q1/Code/Table_SX_Fig2_Hyperparameters_Comprehensive.csv`, `$RESULTS_DIR/R2-q1/Code/benchmark_table_notes.md` and `$RESULTS_DIR/R1-q5/Code/Table_SX_R1q5_MMIL_Comprehensive.csv`: the supplementary tables.
