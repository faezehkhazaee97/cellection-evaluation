# Benchmark runners

One entry script per model family. Each runs the family's training script for one task or all five, then the family's summary script. The scripts are launched one by one with `benchmark_baselines/` as the working directory, and each loops over 10 seeds × 5 folds.

## Scripts

- `run_celection.py`: CELLECTION (`celection/*.py`)
- `run_protocell4p.py`: ProtoCell4P (`protocell4p/*.py`)
- `run_deepset.py`: DeepSet (`deepset/*.py`)
- `run_cloudpred.py`: CloudPred (`cloudpred/*.py`)
- `run_ct_pseudobulk.py`: cell-type pseudobulk baselines (`ct_pseudobulk/*.py`)
- `run_ct_proportion.py`: cell-type proportion baselines (`ct_proportion/*.py`)
- `run_pseudobulk.py`: pseudobulk baselines (`pseudobulk/*.py`)
- `run_raw_script.py`: any script of `benchmark_baselines/`, given by its relative path; extra arguments are forwarded
- `common_runner.py`: the shared launcher behind all of the above

The three linear-baseline runners share one summary script, `ct_pseudobulk/summarize_linear_baselines.py`.

## Options

- `--dataset {all,cardio,cardio_biosample,covid,lupus,lupus_race}` (default `all`)
- `--python /path/to/python`: interpreter for the scripts (default: the current one)
- `--summary` / `--no-summary`: run the family's summary script afterwards (default: on)
- `--dry-run`: print the commands without running them

## How to run

From `experiments-public/benchmark_baselines`, with `DATA_DIR` and `RESULTS_DIR` exported (and `BIOPOINTNET_ROOT` / `CLOUDPRED_PATH` where needed):

```bash
python runners/run_celection.py --dataset all
python runners/run_cloudpred.py --dataset covid
python runners/run_deepset.py --dataset lupus --dry-run
python runners/run_raw_script.py plot/combine_all_test_summaries.py
```
