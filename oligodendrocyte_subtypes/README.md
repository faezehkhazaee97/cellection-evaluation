# Oligodendrocyte subtypes without cross-lineage competition

When CELLECTION sees all four glial classes, oligodendrocytes compete with astrocytes, microglia and OPCs for attention. Rerunning it on SEA-AD MTG oligodendrocytes alone removes that competition and shows which oligodendrocyte supertypes are enriched among the highest-attention cells (Figure S16). `all_glia_rerun.py` repeats the analysis with all four classes, and `train_biopointnet.py` fits a bioPointNet classifier to the same oligodendrocytes. Prompted by Reviewer 1, question 12.

## How to run

```bash
export DATA_DIR=/path/to/Data
export RESULTS_DIR=/path/to/results
export SEAAD_GLIA_DIR=/path/to/SEAAD_Glia          # default: $DATA_DIR/SEAAD_Glia
cd experiments-public/oligodendrocyte_subtypes
python oligodendrocyte_rerun.py                    # oligodendrocytes only, supertype enrichment
python all_glia_rerun.py                           # the same analysis with all four glial classes
python subset_oligodendrocytes.py                  # bioPointNet route: build the oligodendrocyte object,
PYTHONPATH="$BIOPOINTNET_ROOT:$PYTHONPATH" python train_biopointnet.py   # then train (BIOPOINTNET_ROOT = the benchmark_baselines/ folder, which holds bioPointNet_Apr2025.py)
```

What it needs:
- `$DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad` (about 3.7 GB) and sciMultiLaMA cell embeddings under `$SEAAD_GLIA_DIR/MTG/`: `Oligodendrocyte_MTG_sciMultiLaMA_CELL_embedding.csv` for `oligodendrocyte_rerun.py`, `MTG_sciMultiLaMA_CELL_embedding.csv` for `all_glia_rerun.py`, and `sciLaMA_emb_only_correct_sex/Oligodendrocyte_MTG_sciMultiLaMA_CELL_embedding.csv` for `subset_oligodendrocytes.py`. The embeddings come from sciMultiLaMA, which is not included.
- The `cellection` package (`oligodendrocyte_rerun.py`, `all_glia_rerun.py`), bioPointNet (`train_biopointnet.py`), scanpy, torch, scikit-learn and matplotlib.
- A GPU is recommended. The scripts load large expression matrices, so submit them as batch jobs with plenty of memory.
- On one GPU, `oligodendrocyte_rerun.py` takes about 6 min per seed (about 45 min for its 7 seeds) and `all_glia_rerun.py` about 10 min per seed (about 1.5 h for 10). `subset_oligodendrocytes.py` and `train_biopointnet.py` were not timed.

## Files

- `oligodendrocyte_rerun.py`: for seeds 2 and 10-15, trains CELLECTION (AD vs control, one bag per donor) on oligodendrocytes only, compares each supertype's share among the top 100 attention cells per donor with its share among all oligodendrocytes, and draws the enrichment bar plots and UMAPs.
- `all_glia_rerun.py`: the same pipeline on astrocytes, microglia, oligodendrocytes and OPCs together (seeds 6-9 and 15-20).
- `subset_oligodendrocytes.py`: writes the oligodendrocyte-only SEA-AD MTG object, with and without its sciMultiLaMA embedding and UMAP, plus its metadata table, for `train_biopointnet.py`.
- `train_biopointnet.py`: bioPointNet MIL classifier of disease status on that object, 5 seeds, with attention UMAPs.

## Outputs

- `oligodendrocyte_rerun.py` writes to `$RESULTS_DIR/R1-q12/Results/test2/`: `competition_seed_<SEED>.csv` (per donor and major cell type, the share among top-attention cells vs the share among all cells), `oligo_enrichment_all_subtypes_seed_<SEED>.pdf`, `oligo_enrichment_<Supertype>_seed_<SEED>.pdf`, `umap_oligo_subtypes_seed_<SEED>.pdf`, `umap_oligo_top100_seed_<SEED>.pdf`, `umap_embedding_top10_seed_<SEED>_*.pdf`, and model checkpoints in `seed_<SEED>/`.
- `all_glia_rerun.py` writes the same set to `$RESULTS_DIR/R1-q12/Results/test/`, with the figures as `.png`.
- `subset_oligodendrocytes.py` writes `MTG_oligo_sciMultiLaMA_norm.h5ad`, `MTG_oligo_sciMultiLaMA.h5ad` and `MTG_oligo_metadata.csv` to `$RESULTS_DIR/R1-q12/Data/`.
- `train_biopointnet.py` writes to `$RESULTS_DIR/R1-q12/Data/MTG/MTG_cls/`: `CLASSIFIER_MTG_<SEED>.pt`, `ENCODER_MTG_<SEED>.pt`, `RES_STAT_MTG_<SEED>.csv` (training and validation metrics), label tables and heatmaps, `Adata_Tr_MTG_<SEED>.h5ad`, and `figures/umap_*.png`.
