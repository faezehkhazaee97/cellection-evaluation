# Batch correction through conditioning

Can a one-hot batch covariate, fed to CELLECTION at the instance encoder, remove a technical batch effect without also removing the phenotype signal (Reviewer 1, question 1)? To get a known answer, the simulations build synthetic patients that each come from a single technical batch, while the label (one cell type present or absent) occurs in both batches. Two batch pairs are used: pancreas cells from two studies (Baron vs Muraro) and PBMCs from two platforms (10x PBMC 3k vs GSE132044). A third simulation asks whether attention is pulled toward a cell type whose abundance differs by batch but not by label. The top-level scripts repeat the with/without-batch comparison on real data: pancreas samples split into with-beta and no-beta pseudo-samples, and SEA-AD glia with donor sex as the covariate.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
cd experiments-public/batch_correction_simulation/simulations

# Pancreas: label = beta cells present / absent
python simulate_pancreas.py && python evaluate_pancreas.py
python simulate_pancreas_all_celltypes.py && python evaluate_pancreas_all_celltypes.py

# PBMC: label = B cells present / absent (needs pbmc_smartseq_raw.h5ad, see below)
python simulate_pbmc.py && python evaluate_pbmc.py
python simulate_pbmc_all_celltypes.py && python evaluate_pbmc_all_celltypes.py

# Attention test: NK abundance differs by batch but not by label
python simulate_nk_hijack.py
sbatch submit_nk_hijack_test.sbatch          # or: python run_nk_hijack_test.py
```

The PBMC simulations need `pbmc_smartseq_raw.h5ad`, built once from GEO series [GSE132044](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132044):

```bash
mkdir -p "$RESULTS_DIR/R1-q1/Data/pbmc_smartseq" && cd "$RESULTS_DIR/R1-q1/Data/pbmc_smartseq"
# download and gunzip GSE132044_pbmc_hg38_count_matrix.mtx.gz, GSE132044_pbmc_hg38_gene.tsv.gz, GSE132044_pbmc_hg38_cell.tsv.gz
python /path/to/experiments-public/batch_correction_simulation/build_gse132044_h5ad.py   # reads and writes in the current folder
```

The real-data scripts at the top level of `batch_correction_simulation/` run from that folder, for example `python pancreas_pseudosamples.py`, `python seaad_with_batch.py` or `python check_batch_balance.py --split`.

Input data:
- `$DATA_DIR/pancreatic_w_batch/Fig2_obj.h5ad`: multi-study human pancreas, with the study in `obs["dataset"]` and the cell type in `obs["celltype"]`.
- `$DATA_DIR/pbmc3k/pbmc3k_raw.h5ad`: 10x PBMC 3k raw counts.
- `$RESULTS_DIR/R1-q1/Data/pbmc_smartseq/pbmc_smartseq_raw.h5ad`: built above. It holds the whole GSE132044 `pbmc_hg38` matrix, all protocols, and the scripts call it `smartseq`.
- `$DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad`: for the two SEA-AD scripts and `check_batch_balance.py`.

The scripts import the `cellection` package, scanpy (with leidenalg), anndata, torch, scikit-learn, matplotlib and scib (ASW metrics); the SEA-AD scripts also import seaborn. Training scripts (`evaluate_*.py`, `run_nk_hijack_test.py`, `pancreas_pseudosamples*.py`, `seaad_with*_batch.py`) use a GPU if there is one and fall back to CPU. The simulation builders and the two check scripts are CPU work. On one GPU, a 10-seed `evaluate_*.py` run (with and without batch) takes about 10-15 min, each simulation builder a few minutes, the attention test about 30 min, and `pancreas_pseudosamples.py` about 1 h.

## Files

- `build_gse132044_h5ad.py`: turns the GSE132044 count matrix, gene list and cell list in the current folder into `pbmc_smartseq_raw.h5ad`.
- `pancreas_pseudosamples.py`: splits each real sample of every study in `Fig2_obj.h5ad` into a with-beta and a no-beta pseudo-sample, then trains CELLECTION with and without the study as batch covariate (seeds 0-4); writes metrics, models and UMAPs to `Results_pancreas/`.
- `pancreas_pseudosamples_lab_counts.py`: the same run plus per-study cell-count tables, written to `Results_pancreas_new/`.
- `seaad_with_batch.py`: SEA-AD MTG glia, AD vs control, with `sex` as batch covariate (80/20 donor split, seeds 10-14); writes `Results/withBatch/`.
- `seaad_without_batch.py`: the same analysis on the same donor splits, without a covariate; writes `Results/WithoutBatch/`.
- `check_batch_balance.py`: donor-level batch (default `sex`) × disease tables with chi-square and Cramér's V for the SEA-AD data; stdout only.
- `check_pseudosample_eligibility.py`: counts the pancreas samples with at least one beta and one non-beta cell, the ones the pseudo-sample split keeps; stdout only.
- `simulations/simulate_pancreas.py`: pancreas cohort of 500-cell bags, each drawn from one study (Baron or Muraro) using its alpha, beta, delta and gamma cells; 20 bags per batch, half without beta cells. Writes the h5ad and composition tables.
- `simulations/simulate_pancreas_all_celltypes.py`: the pancreas cohort with every cell type of each study and 10 bags per batch.
- `simulations/evaluate_pancreas.py`: trains CELLECTION with and without batch conditioning on the pancreas cohort (seeds 5-14, fixed held-out test bags) and writes bag-level and cell-level metrics, models and UMAPs.
- `simulations/evaluate_pancreas_all_celltypes.py`: the same evaluation on the all-cell-type pancreas cohort.
- `simulations/evaluate_pancreas_single_seed.py`: one seed (8) of the pancreas evaluation, with 8 of 10 bags per batch and label held out and cell-level metrics on the test cells; writes to `Results_pancreas_synth_cellect_test/`.
- `simulations/simulate_pbmc.py`: PBMC cohort. Annotates B / T / Mono / NK by joint Leiden clustering and marker genes, then draws 500-cell bags from one platform with multinomial cell-type counts, 20 bags per batch.
- `simulations/simulate_pbmc_all_celltypes.py`: the PBMC cohort with fixed, equal cell-type counts and 10 bags per batch. `simulate_nk_hijack.py` imports its `load_and_align()`.
- `simulations/evaluate_pbmc.py`: the PBMC counterpart of `evaluate_pancreas.py` (seeds 5-14, same metrics).
- `simulations/evaluate_pbmc_all_celltypes.py`: the same evaluation on the fixed-count PBMC cohort.
- `simulations/simulate_nk_hijack.py`: PBMC bags in which NK cells are 5% of non-B cells on one platform and 30% on the other, while the label depends only on B cells. Runs on CPU.
- `simulations/run_nk_hijack_test.py`: trains a batch-blind CELLECTION model on those bags (seeds 3, 4, 6, 7, 8) and checks whether high-attention cells are enriched for NK cells on each platform.
- `simulations/submit_nk_hijack_test.sbatch`: Slurm GPU job for `run_nk_hijack_test.py`. Set `--account`, `--partition` and `VENV_DIR`, then submit from `simulations/` with `DATA_DIR` and `RESULTS_DIR` exported.

## Outputs

Under `$RESULTS_DIR/R1-q1/` unless noted.
- Simulated cohorts, under `$DATA_DIR`: `pancreatic_w_batch/pancreas_synthetic_baron_muraro[_all_celltypes].h5ad`, `pbmc3k/pbmc_real_10x_vs_smartseq[_all_celltypes].h5ad`, `pbmc3k/pbmc_batch_confounded_nk_hijack.h5ad`.
- `Results_pancreas_baron_muraro_synth[_all_celltypes]/`, `Results_pbmc_real_batch[_all_celltypes]/`: bag and cell-type composition tables (`cells_per_batch*.csv`, `bag_counts_per_batch.csv`, `bag_sizes.csv`, `cells_per_sample_by_celltype.csv`, and `sample_metadata.csv` for PBMC) and pre-training UMAPs coloured by batch, label and cell type.
- `Results_pancreas_synth_cellect[_all_celltypes]/`, `Results_pbmc_real_batch_eval[_all_celltypes]/`:
  - `classification_metrics.csv`: bag-level AUROC, AUPRC, macro-F1 and per-class F1 on the test bags, one row per seed × condition (`no_batch`, `with_batch`);
  - `embedding_metrics.csv`: cell-level ARI and NMI (Leiden vs cell type), cell-type ASW and batch ASW of the instance-encoder embeddings, one row per seed × condition;
  - per-run copies `*_<condition>_seed_<s>.csv`, held-out bag IDs `test_sample_ids.csv`, models `models_<condition>_seed_<s>/`, and UMAPs `umap_<condition>_{batch,label,celltype}.png`.
- `Results_pancreas_synth_cellect_test/`: the single-seed run, with the same metric files (one row per condition), `test_sample_ids.csv`, models and `with_batch` UMAPs.
- `Results_attention_hijack_test/`: from the simulation, `sample_composition.csv`, `celltype_counts_by_batch.csv` and `celltype_frac_by_batch.csv`; from the test, `attn_df_seed_<s>.csv` (per-cell attention), `per_patient_hijack_metrics.csv`, `nk_hijack_summary_by_batch.csv`, `nk_hijack_foldchange_by_batch.csv` and `models_seed_<s>/`. The Slurm log goes to the submit folder as `slurm_attn_hijack_<jobid>.out`.
- `Results_pancreas/`, `Results_pancreas_new/`: metrics, models and UMAPs from the two pseudo-sample scripts, plus the cell-count tables in `Results_pancreas_new/`.
- `Results/withBatch/metrics_with_batch_5seeds.csv`, `Results/WithoutBatch/metrics_no_batch_5seeds.csv`: per-seed AUROC and macro-F1 on held-out donors, next to the `seed_<s>/` model folders.
