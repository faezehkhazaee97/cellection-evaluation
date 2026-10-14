# Identifying a planted subpopulation

If one cell type drives the phenotype, does CELLECTION's attention point at it? We build simulated cohorts from real PBMC data in which B-cell abundance is the only difference between cases and controls, train CELLECTION on them, and check whether the highest-attention cells are the planted B cells. This answers Reviewer 1, question 4.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/subpopulation_identification
python simulate_pbmc3k.py            && python identify_subpopulations_pbmc3k.py
python simulate_pbmc10k.py           && python identify_subpopulations_pbmc10k.py
python simulate_pbmc33k.py           && python identify_subpopulations_pbmc33k.py
python simulate_pbmc3k_control0b.py  && python identify_subpopulations_pbmc3k_control0b.py   # controls with no B cells
python simulate_pbmc68k.py                                                                   # PBMC68k cohort only
```

The simulations start from 10x PBMC raw counts saved as h5ad: `$DATA_DIR/pbmc3k/pbmc3k_raw.h5ad`, `$DATA_DIR/pbmc10k/pbmc10k_raw.h5ad`, `$DATA_DIR/pbmc33k/pbmc33k_raw.h5ad` and `$DATA_DIR/pbmc68k/pbmc68k_raw.h5ad`. Only PBMC3k is downloaded automatically (by scanpy); the others have to be in place. Besides the `cellection` package you need scanpy (with igraph and leidenalg for Leiden clustering; without them the scripts fall back to KMeans), torch, scikit-learn, seaborn and h5py.

The `simulate_*.py` scripts run on CPU. The PBMC3k cohort takes minutes to build and the larger ones longer. The `identify_subpopulations_*.py` scripts use a GPU when there is one: on one GPU, PBMC3k takes about 30 min per seed (about 2.5 h for its 5 seeds) and the PBMC10k script about 3 h for its 4 seeds.

## Files

- `simulate_pbmc3k.py`: clusters PBMC3k, names each cluster B, T, Mono or NK from one marker gene per type, and samples 50 patients of 1,000 cells with replacement, cases with 30-40% B cells and controls with 0-5%. Writes the cohort h5ad and two sanity tables.
- `simulate_pbmc10k.py`: the same construction on PBMC10k, 40 patients of 500 cells.
- `simulate_pbmc33k.py`: the same on PBMC33k, 50 patients of 1,000 cells.
- `simulate_pbmc68k.py`: the same on PBMC68k, 50 patients of 1,000 cells. No script here trains on this cohort.
- `simulate_pbmc3k_control0b.py`: the PBMC3k cohort with every control at exactly 0% B cells.
- `identify_subpopulations_pbmc3k.py`: trains binary CELLECTION on the PBMC3k cohort with seeds 6, 7, 4, 8 and 3, reports patient-level AUROC/AUPRC/macro-F1, and measures how strongly the top-attention cells are enriched for B cells (cell AUPRC, Enrich@1/5/10%, fold enrichment over the bag's B fraction). Also draws UMAPs of the top-attention cells.
- `identify_subpopulations_pbmc10k.py`: the PBMC10k cohort with seeds 4, 6, 8 and 3; cell metrics only on correctly classified patients, plus best-seed and across-seed plots.
- `identify_subpopulations_pbmc33k.py`: the PBMC33k cohort, seed 4.
- `identify_subpopulations_pbmc3k_control0b.py`: the cohort whose controls have no B cells, seed 3.

## Outputs

- Cohorts go to `$DATA_DIR/pbmc<N>/`: `pbmc<N>_processed_with_celltypes.h5ad` (cells with B/T/Mono/NK labels), `pbmc<N>_simulated_patients_Bcell_causal.h5ad` (the cohort; `pbmc3k_simulated_patients_Bcell_control0pct.h5ad` for the 0%-B controls) and two sanity tables, `..._sanity_by_patient.csv` (B fraction and cell count per patient) and `..._sanity_summary.csv` (one row per cohort). `simulate_pbmc3k_control0b.py` writes its processed h5ad to the same path as `simulate_pbmc3k.py` and overwrites it.
- `identify_subpopulations_pbmc3k.py` writes to `$RESULTS_DIR/R1-q4a/Results/`:
  - `patient_level_metrics.csv`: AUROC, AUPRC and macro-F1 per seed;
  - `cell_level_metrics.csv`: CellAUPRC, Enrich@1/5/10%, BackgroundB and fold enrichment per seed;
  - `dataset_case_control_sanity.csv`, `umap_panel_sanity_counts.csv`: cohort and plot checks;
  - `enrichment_barplot.pdf`, `umap_cell_types.pdf`, `seed_<S>_umap_top10_*.pdf`, and model checkpoints in `seed_<S>/`.
- The PBMC10k, PBMC33k and 0%-B scripts write to `$RESULTS_DIR/R1-q4a/Results/test_10k/`, `test_33k/` and `test_control0B/`. Each writes `patient_level_metrics.csv`, `cell_level_metrics.csv` (with an extra `n_patients_eval` column for PBMC10k and PBMC33k), `enrichment_barplot.pdf`, `umap_cell_types.pdf`, per-seed UMAPs and checkpoints as above. In place of the two sanity tables there is `case_control_subset_checks.csv`, and there are two more plots: `umap_case_control.pdf` and `seed_<S>_umap_top10_{B,nonB}.pdf`. PBMC10k and PBMC33k also write `enrichment_barplot_best_seed.pdf`, and PBMC10k writes `enrichment_across_seeds.{pdf,png}`.
