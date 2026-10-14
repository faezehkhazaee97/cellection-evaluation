# CELLECTION: additional experiments

Analyses added to the CELLECTION manuscript during peer review: benchmarks against other methods, ablations of parts of the model, simulations with a known answer, and checks on how results depend on sample size and cell counts. Each folder is one experiment and has its own README.

The CELLECTION method and package were developed by Hongru Hu and are available at [quon-titative-biology/CELLECTION](https://github.com/quon-titative-biology/CELLECTION).

| Folder | What it tests | Reviewer question |
|---|---|---|
| [batch_correction_simulation](batch_correction_simulation/) | Whether a one-hot batch covariate at the instance encoder removes batch effects, on simulated pancreas (Baron vs Muraro) and PBMC (10x vs GSE132044) cohorts. Also checks whether attention follows a cell type whose abundance differs by batch but not by label. | R1 Q1 |
| [tnet_ablation](tnet_ablation/) | What the two T-Net modules add, with every on/off combination on five classification tasks (Figure S2). | R1 Q2 |
| [seaad_scilama_vs_raw](seaad_scilama_vs_raw/) | Raw expression vs sciMultiLaMA embeddings as input, for AD vs control in SEA-AD MTG glia, and the training time of each. | R1 Q3 |
| [subpopulation_identification](subpopulation_identification/) | Whether attention picks out B cells when B-cell abundance is the only difference between simulated cases and controls. | R1 Q4 |
| [pancreas_batch_mixing](pancreas_batch_mixing/) | How a batch covariate changes batch mixing and cell-type separation of per-cell representations, with pancreas studies as batches. | R1 Q4 |
| [mmil_baseline](mmil_baseline/) | A cell-level ridge logistic regression (MMIL-style) baseline on the three binary benchmark tasks. | R1 Q5 |
| [cell_count_imbalance](cell_count_imbalance/) | Whether accuracy depends on cells per sample, before and after downsampling the largest samples (Figure S4). | R1 Q8 |
| [min_patients_covid](min_patients_covid/) | How performance changes with 5 to 80 training patients on Haniffa COVID-19 severity. | R1 Q10 |
| [ordinal_cutpoints](ordinal_cutpoints/) | Learned vs fixed cut points in the ordinal loss (Figure S10). | R1 Q11 |
| [oligodendrocyte_subtypes](oligodendrocyte_subtypes/) | Which oligodendrocyte supertypes come up when CELLECTION is rerun on SEA-AD oligodendrocytes alone (Figure S16). | R1 Q12 |
| [compute_scaling](compute_scaling/) | Training time and peak GPU memory as patients and cells per patient increase (Figure S6). | R1 Q13 |
| [benchmark_baselines](benchmark_baselines/) | CELLECTION against ProtoCell4P, DeepSet, CloudPred and three linear baselines on five tasks (Figure 2, Table S3). | R2 Q1, Q2, Q6 |
| [seaad_donor_attribution](seaad_donor_attribution/) | How each SEA-AD donor's predicted disease probability relates to where attention falls among glial classes (Figure S15). | R2 Q3 |
| [attention_ablation](attention_ablation/) | Whether removing the highest-attention cells hurts prediction more than removing random cells (Figure S7). | R2 Q4 |
| [hwi_regression_stats](hwi_regression_stats/) | Correlations, p-values and bootstrap intervals for the hand-wing index regression (Figure S18). | R2 Q5 |

## Setup

All paths come from environment variables. Give absolute paths, because some runners change the working directory.

```bash
export DATA_DIR=/abs/path/to/Data          # input datasets
export RESULTS_DIR=/abs/path/to/results    # outputs
```

| Variable | Used by | Points to |
|---|---|---|
| `DATA_DIR` | almost everything | Input datasets: `lupus/`, `covid/`, `cardio/`, `pbmc3k/`, `pbmc10k/`, `pbmc33k/`, `pancreatic_w_batch/`, `sea-ad/`, `haniffa21_intersection.h5ad` and so on. |
| `RESULTS_DIR` | almost everything | Output root. Each experiment writes to its own subfolder, named in its README (`tnet_ablation` writes to `$RESULTS_DIR/R1-q2/Results/`, for example). A few experiments also read intermediate files from here: `R1-q1/Data/`, `R1-q12/Data/`, `R2-q5/Data/`. |
| `SEAAD_GLIA_DIR` | oligodendrocyte_subtypes | SEA-AD glia embeddings (`MTG/...`). Defaults to `$DATA_DIR/SEAAD_Glia`. |
| `BIOPOINTNET_ROOT` | benchmark_baselines/celection, oligodendrocyte_subtypes | Folder with `bioPointNet_Apr2025.py`, the CELLECTION model code used for the Figure 2 runs. The file is in `benchmark_baselines/`, so point this at that folder. |
| `CLOUDPRED_PATH` | benchmark_baselines/cloudpred | A clone of https://github.com/bryanhe/CloudPred, if the `cloudpred` package is not installed. |

Unset, `DATA_DIR` and `RESULTS_DIR` fall back to `data` and `results` in the current directory.

Install the Python dependencies from the repository root, at the versions the experiments ran with:

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CELLECTION_CG_DEV=0.1.dev14+g16d2c14f0
pip install -r requirements.txt
```

The `export` gives the vendored `cellection` package its version number, which setuptools-scm cannot read from git inside this repository. `environment.yml` adds R for `mmil_baseline` and installs `requirements.txt` through pip, so set the same `export` before creating it.

The results in the paper were produced with the `cellection` package at fork commit `16d2c14`, included unchanged in `third_party/cellection/` under its original MIT license. The clone instruction in `third_party/cellection/README.md` points to a private repository and can be ignored; install from `third_party/cellection/` instead.

Most experiments import the `cellection` package. Training scripts use a GPU if there is one and run on CPU otherwise, except `compute_scaling`, which measures GPU memory and needs CUDA. On a shared cluster, submit the training scripts as batch jobs.

## Data

Apart from the fold assignments in `folds/`, the input data is not in this repository. Put each file under `$DATA_DIR` at the path shown.

| Path under `$DATA_DIR` | Dataset | Source paper |
|---|---|---|
| `covid/covid_Ziegler_processed.h5ad` | Nasopharyngeal swabs from 58 COVID-19 and control participants | Ziegler et al. 2021, *Cell*, [doi:10.1016/j.cell.2021.07.023](https://doi.org/10.1016/j.cell.2021.07.023) |
| `cardio/cardio_processed.h5ad` | Left ventricle, dilated (DCM) and hypertrophic (HCM) cardiomyopathy and non-failing (NF) hearts | Chaffin et al. 2022, *Nature*, [doi:10.1038/s41586-022-04817-8](https://doi.org/10.1038/s41586-022-04817-8) |
| `lupus/lupus_processed.h5ad` | PBMCs, SLE vs healthy, Asian and White donors | Perez et al. 2022, *Science*, [doi:10.1126/science.abf1970](https://doi.org/10.1126/science.abf1970) |
| `haniffa21_intersection.h5ad` | COVID-19 PBMCs with worst clinical status per patient | Stephenson et al. 2021, *Nature Medicine*, [doi:10.1038/s41591-021-01329-2](https://doi.org/10.1038/s41591-021-01329-2) |
| `sea-ad/MTG_glia_intersection_norm.h5ad` | SEA-AD middle temporal gyrus, glial cells | Gabitto et al. 2024, *Nature Neuroscience*, [doi:10.1038/s41593-024-01774-5](https://doi.org/10.1038/s41593-024-01774-5) |
| `SEAAD_Glia/MTG/` | sciMultiLaMA cell embeddings of the SEA-AD glia | Not included |
| `pancreatic_w_batch/Fig2_obj.h5ad` | Human pancreas, five studies as batches (`obs["dataset"]`) | Baron et al. 2016, *Cell Systems*, [doi:10.1016/j.cels.2016.08.011](https://doi.org/10.1016/j.cels.2016.08.011); Muraro et al. 2016, *Cell Systems*, [doi:10.1016/j.cels.2016.09.002](https://doi.org/10.1016/j.cels.2016.09.002); Segerstolpe et al. 2016, *Cell Metabolism*, [doi:10.1016/j.cmet.2016.08.020](https://doi.org/10.1016/j.cmet.2016.08.020); Wang et al. 2016, *Diabetes*, [doi:10.2337/db16-0405](https://doi.org/10.2337/db16-0405); Xin et al. 2016, *Cell Metabolism*, [doi:10.1016/j.cmet.2016.08.018](https://doi.org/10.1016/j.cmet.2016.08.018) |
| `pbmc3k/pbmc3k_raw.h5ad` | 10x Genomics PBMC 3k | Downloaded by scanpy on first run |
| `pbmc10k/`, `pbmc33k/`, `pbmc68k/` | 10x Genomics PBMC 10k, 33k and 68k | 10x Genomics |

`folds/` holds the 5-fold assignments for the COVID, Cardio and lupus tasks (label, `samples`, `fold`) in the same layout as `$DATA_DIR`; copy them in with `cp -r folds/* "$DATA_DIR"/`.

Built from the sources above by scripts in this repository: `$RESULTS_DIR/R1-q1/Data/pbmc_smartseq/` (GEO GSE132044, Ding et al. 2020, *Nature Biotechnology*, [doi:10.1038/s41587-020-0465-8](https://doi.org/10.1038/s41587-020-0465-8), by `batch_correction_simulation/build_gse132044_h5ad.py`) and `$RESULTS_DIR/R1-q12/Data/` (by `oligodendrocyte_subtypes/subset_oligodendrocytes.py`). `$RESULTS_DIR/R2-q5/Data/` holds the hand-wing index predictions read by `hwi_regression_stats`; they come from a CELLECTION regression run that is not part of this repository.

## Figures and tables

| Figure / table | Scripts, in run order |
|---|---|
| Figure 2, Table S3 | `benchmark_baselines/`: the per-model training scripts, then each model's `summarize*.py`, `plot/combine_all_test_summaries.py`, `plot/bubbleplot_from_combined.py` (commands in `REPRODUCIBILITY_FIG2.md`) |
| Supplementary benchmark tables | `benchmark_baselines/scripts/generate_comprehensive_benchmark_tables.py` |
| Figure S2 | `tnet_ablation/four_configs/<task>.py`, then `tnet_ablation/plot_tnet_bars.py` |
| Figure S4 | `cell_count_imbalance/<task>/before_downsampling.py` and `downsample_and_retrain.py` (`lupus_race/before_and_after_downsampling.py`), then `<task>/plot_before_after.py` |
| Figure S6 | `compute_scaling/covid_scaling.py` |
| Figure S7 | `attention_ablation/<task>.py` (`covid.py` is panel a) |
| Figure S10 | `ordinal_cutpoints/learned_cutpoints.py` and `fixed_cutpoints.py`, then `plot_spearman.py` |
| Figure S15 | `seaad_donor_attribution/donor_attribution.py` |
| Figure S16 | `oligodendrocyte_subtypes/` |
| Figure S18 | `hwi_regression_stats/correlation_metrics.py` |

## Not included

- Input data, apart from the fold files (see Data).
- Results, figures and model checkpoints.
- The supplementary tables. `benchmark_baselines/scripts/generate_comprehensive_benchmark_tables.py` writes them.
- External code: CloudPred (https://github.com/bryanhe/CloudPred). ProtoCell4P's original implementation is at https://github.com/gzxiong/ProtoCell4P; the `ProtoCell` model class the benchmark uses is defined in each `benchmark_baselines/protocell4p/*.py`.

## License

MIT; see `LICENSE`. `third_party/cellection/` keeps its own MIT license, in `third_party/cellection/LICENSE`.
