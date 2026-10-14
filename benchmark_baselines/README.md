# Benchmark against existing methods

How does CELLECTION compare with existing sample-level classifiers and with simple linear baselines? This folder trains CELLECTION, ProtoCell4P, DeepSet, CloudPred and three linear baselines (cell-type pseudobulk, cell-type proportions, pseudobulk) on five tasks (Ziegler COVID-19, lupus SLE vs healthy, lupus ancestry, Cardio by donor and by biosample) and reports AUROC, macro-F1 and macro AUPRC, which make up Figure 2 and Table S3. It was set up for Reviewer 2's questions 1, 2 and 6.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
export BIOPOINTNET_ROOT=/abs/path/to/experiments-public/benchmark_baselines   # folder with bioPointNet_Apr2025.py
export CLOUDPRED_PATH=/path/to/CloudPred           # clone of github.com/bryanhe/CloudPred, if cloudpred is not installed
cd experiments-public/benchmark_baselines

bash scripts/run_all_benchmark_experiments.sh      # every model on every task, then summaries, table and plots
bash scripts/reproduce_fig2_from_results.sh        # summaries, table and plots only, from existing results
python runners/run_celection.py --dataset lupus    # one model family, one task (see runners/README.md)
python celection/lupus.py                          # or any training script directly
```

By hand, run the training scripts first, then each family's `summarize*.py`, then `plot/combine_all_test_summaries.py` and finally `plot/bubbleplot_from_combined.py`. ProtoCell4P's prototype count is the `n_proto` variable in each script (8 for the two lupus tasks, 16 for the rest); Figure 2 shows 4, 8 and 16, and each count writes to its own `proto_<n>/` folder.

Data, relative to `$DATA_DIR`: `lupus/lupus_processed.h5ad`, `lupus/lupus_5folds.csv`, `lupus/race_5folds.csv`, `covid/covid_Ziegler_processed.h5ad`, `covid/covid_5folds.csv`, `cardio/cardio_processed.h5ad`, `cardio/cardio_5folds.csv` and `cardio/cardio_biosample_id_5folds.csv`. Every folds file has the sample label as its index column, then `samples` and `fold` (1-5).

Code: `bioPointNet_Apr2025.py` (in this folder) for `celection/`, the `cloudpred` package for `deepset/` and `cloudpred/`, plus scanpy/anndata, torch, scikit-learn, scipy, pandas, matplotlib/seaborn and tqdm. CloudPred is at https://github.com/bryanhe/CloudPred. ProtoCell4P's original implementation is at https://github.com/gzxiong/ProtoCell4P (Xiong et al. 2023, *Bioinformatics*); the `ProtoCell` model class used here is defined in each `protocell4p/*.py`.

Every training script loads the whole h5ad into memory. CELLECTION and ProtoCell4P use a GPU when there is one. On one GPU the largest task (lupus) takes about 7.5 h for CELLECTION, 6 h per prototype count for ProtoCell4P, 3 h for DeepSet and up to 20 h for CloudPred; COVID is much quicker (CELLECTION about 20 min). The linear baselines were not timed. Rebuilding the table and plots from existing results takes seconds.

## Files

Each training script runs one model on one task for 10 seeds × 5 folds; the first script of each family below says what the family does, the rest name the task. In the three linear-baseline folders, `lupus.py` carries the full protocol in its docstring and its siblings list only what differs.

- `bioPointNet_Apr2025.py`: the CELLECTION model code that `celection/*.py` imports; `BIOPOINTNET_ROOT` points here.
- `celection/covid.py`: CELLECTION on Ziegler COVID-19, case vs control; writes per-run metrics, predictions and checkpoints.
- `celection/lupus.py`: lupus, SLE vs healthy.
- `celection/lupus_race.py`: lupus ancestry, Asian vs White (`pop_cov`).
- `celection/cardio.py`: Cardio (DCM / HCM / NF), one bag per donor.
- `celection/cardio_biosample.py`: Cardio, one bag per biosample.
- `celection/summarize.py`: mean CELLECTION test metrics over the 50 runs per task, into `celection_test_summary.csv`.
- `protocell4p/covid.py`: ProtoCell4P (prototype count `n_proto`, set in the script) on Ziegler COVID-19.
- `protocell4p/lupus.py`: lupus, SLE vs healthy.
- `protocell4p/lupus_race.py`: lupus ancestry.
- `protocell4p/cardio.py`: Cardio, per donor.
- `protocell4p/cardio_biosample.py`: Cardio, per biosample.
- `protocell4p/summarize.py`: mean test metrics per task and prototype count, into `protocell4p_test_summary.csv`.
- `deepset/covid.py`: DeepSet on Ziegler COVID-19, once on raw genes and once on PCA features.
- `deepset/lupus.py`: lupus, SLE vs healthy.
- `deepset/lupus_race.py`: lupus ancestry.
- `deepset/cardio.py`: Cardio, per donor.
- `deepset/cardio_biosample.py`: Cardio, per biosample.
- `deepset/summarize.py`: raw-gene test metrics only, into `deepset_test_summary.csv`.
- `cloudpred/covid.py`: CloudPred on Ziegler COVID-19, raw genes and PCA features.
- `cloudpred/lupus.py`: lupus, SLE vs healthy.
- `cloudpred/lupus_race.py`: lupus ancestry.
- `cloudpred/cardio.py`: Cardio, per donor.
- `cloudpred/cardio_biosample.py`: Cardio, per biosample.
- `cloudpred/summarize.py`: raw-gene test metrics only, into `cloudpred_test_summary.csv`.
- `ct_pseudobulk/covid.py`: random forest and one-vs-rest logistic regression on per-cell-type pseudobulk vectors, Ziegler COVID-19.
- `ct_pseudobulk/lupus.py`: lupus, SLE vs healthy.
- `ct_pseudobulk/lupus_race.py`: lupus ancestry.
- `ct_pseudobulk/cardio.py`: Cardio, per donor.
- `ct_pseudobulk/cardio_biosample.py`: Cardio, per biosample; logistic regression only.
- `ct_pseudobulk/summarize_linear_baselines.py`: logistic-regression test summaries of all three linear baselines (`ct_pseudobulk/`, `ct_proportion/`, `pseudobulk/`).
- `ct_proportion/covid.py`: the same two models on per-sample cell-type proportions, Ziegler COVID-19.
- `ct_proportion/lupus.py`: lupus, SLE vs healthy.
- `ct_proportion/lupus_race.py`: lupus ancestry.
- `ct_proportion/cardio.py`: Cardio, per donor.
- `ct_proportion/cardio_biosample.py`: Cardio, per biosample (both models).
- `pseudobulk/covid.py`: the same two models on one pseudobulk profile per sample, Ziegler COVID-19.
- `pseudobulk/lupus.py`: lupus, SLE vs healthy.
- `pseudobulk/lupus_race.py`: lupus ancestry.
- `pseudobulk/cardio.py`: Cardio, per donor.
- `pseudobulk/cardio_biosample.py`: Cardio, per biosample; logistic regression only.
- `plot/combine_all_test_summaries.py`: merges the seven per-model summaries into `all_models_test_summary.csv` (the Figure 2 / Table S3 values).
- `plot/bubbleplot_from_combined.py`: Figure 2 bubble and mean-rank bar plots from `all_models_test_summary.csv`.
- `plot/bubbleplot_from_summaries.py`: the same plots straight from the seven per-model summaries.
- `scripts/run_all_benchmark_experiments.sh`: every family on every task through the runners, then summaries, table and plots.
- `scripts/reproduce_fig2_from_results.sh`: summaries, table and plots only.
- `scripts/generate_comprehensive_benchmark_tables.py`: supplementary hyper-parameter tables for the benchmark and the MMIL baseline.
- `runners/README.md`: runner options and examples.
- `runners/common_runner.py`: maps model and task to a script, launches it, then runs the family's summary.
- `runners/run_celection.py`: CELLECTION scripts for one task or all, then their summary.
- `runners/run_protocell4p.py`: the same for ProtoCell4P.
- `runners/run_deepset.py`: the same for DeepSet.
- `runners/run_cloudpred.py`: the same for CloudPred.
- `runners/run_ct_pseudobulk.py`: the same for the cell-type pseudobulk baselines.
- `runners/run_ct_proportion.py`: the same for the cell-type proportion baselines.
- `runners/run_pseudobulk.py`: the same for the pseudobulk baselines.
- `runners/run_raw_script.py`: any script of this folder through the same launcher, with extra arguments forwarded.
- `REPRODUCIBILITY_FIG2.md`: commands that rebuild Figure 2, Table S3 and the supplementary tables.
- `standalone/cloudpred_cardio.py`: a self-contained CloudPred run on Cardio (donor bags, raw genes); per-run metrics go to `./data/cardio_cls/`.
- `standalone/combine_cloudpred_metrics.py`: averages CloudPred test metrics per dataset folder of `$RESULTS_DIR/R2-q1/Results/CloudPred/`.
- `standalone/ranking_plots.py`: improvement percentages, Wilcoxon and t-tests, and rank plots from `$RESULTS_DIR/R2-q1/Results/Plots/{auROC,macroF1,macroAUPRC}.csv` (tables no script here writes).

## Outputs

The benchmark writes under `$RESULTS_DIR/R2-q1/Results/`. Task folders are `Covid`, `Lupus` (`Lupus_Disease` for the linear baselines), `Lupus_Race` (`LupusRace` for DeepSet and CloudPred), `Cardio` and `CardioBiosample`.

- `CELECTION/<Task>/`: best checkpoints, per-run metric CSVs, `*_results_all_folds.pth` and `*_metrics_all.csv` (train, validation and test macro AUROC, macro-F1 and macro AUPRC for all 50 runs), summarised in `CELECTION/celection_test_summary.csv`.
- `4_ProtoCell4P/<Task>/proto_<n>/`: the same files per prototype count, summarised in `4_ProtoCell4P/protocell4p_test_summary.csv`.
- `2_DeepSet/<Task>/`, `cloudpred/<Task>/`: per-sample `.npz` inputs under `data/`, per-run results for raw genes and for PCA features (`PC_` prefix), `*_metrics_all.csv`, summarised in `2_DeepSet/deepset_test_summary.csv` and `cloudpred/cloudpred_test_summary.csv`.
- `ct_pb/<Task>/`, `proportion/<Task>/`, `pseudobulk/<Task>/`: `LogisticRegression_*` and `RF_*` per-run CSVs (no RF runs for `CardioBiosample` in `ct_pb/` and `pseudobulk/`), `*_res_stat_dict_*.pth`, `*_results_mean.csv`, `*_metrics_all.csv` and `logreg_test_summary.csv`; one `<folder>_logreg_results_mean_summary.csv` per baseline folder.
- `all_models_test_summary.csv`: mean test AUROC, macro-F1 and macro AUPRC per model and task (Table S3).
- `plot/{auROC,macroF1,AUPRC}_bubble.pdf` and `plot/*_mean_rankings_barplot.pdf`: the Figure 2 panels.

Elsewhere: `scripts/generate_comprehensive_benchmark_tables.py` writes `$RESULTS_DIR/R2-q1/Code/Table_SX_Fig2_Hyperparameters_Comprehensive.csv`, `$RESULTS_DIR/R2-q1/Code/benchmark_table_notes.md` and `$RESULTS_DIR/R1-q5/Code/Table_SX_R1q5_MMIL_Comprehensive.csv`. Of the `standalone/` scripts, `cloudpred_cardio.py` writes to `./data/cardio_cls/` in the folder it runs from (it also creates an empty `$RESULTS_DIR/R2-q1/Results/CloudPred/Cardio/full_metrics/`), `combine_cloudpred_metrics.py` writes `combined_metrics.csv` and one CSV per metric into `$RESULTS_DIR/R2-q1/Results/CloudPred/`, and `ranking_plots.py` writes PNGs to `$RESULTS_DIR/R2-q1/Code/Plots/Results/`.
