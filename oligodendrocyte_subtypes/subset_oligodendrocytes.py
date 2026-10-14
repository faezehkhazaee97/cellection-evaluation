"""Builds the oligodendrocyte-only SEA-AD MTG object that train_biopointnet.py trains on.

From the normalized MTG glia object it keeps obs["Subclass"] == "Oligodendrocyte", switches gene
names to symbols (var["feature_name"]), stores the unfiltered matrix in .raw and drops genes seen in
fewer than 5 cells. The oligodendrocyte sciMultiLaMA embedding (sciLaMA_emb_only_correct_sex
version, one row per cell indexed by cell id) goes into obsm["X_sciMultiLaMA"], and a 25-NN graph
and UMAP are computed on it.

Reads $DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad and
$SEAAD_GLIA_DIR/MTG/sciLaMA_emb_only_correct_sex/Oligodendrocyte_MTG_sciMultiLaMA_CELL_embedding.csv.
Writes to $RESULTS_DIR/R1-q12/Data/:
  MTG_oligo_sciMultiLaMA_norm.h5ad  the subset, saved before the embedding and UMAP are added
  MTG_oligo_metadata.csv            its obs table (read by train_biopointnet.py)
  MTG_oligo_sciMultiLaMA.h5ad       with embedding, neighbors and UMAP (also read there)
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root
SEAAD_GLIA_DIR = os.environ.get("SEAAD_GLIA_DIR", os.path.join(DATA_DIR, "SEAAD_Glia"))  # embedding CSVs


import pandas as pd
import scanpy as sc

data_path = f'{DATA_DIR}/sea-ad/'
emb_path = f'{SEAAD_GLIA_DIR}/MTG/sciLaMA_emb_only_correct_sex/'
output_dir = f'{RESULTS_DIR}/R1-q12/Data'

os.makedirs(output_dir, exist_ok=True)

adata = sc.read_h5ad(os.path.join(data_path, 'MTG_glia_intersection_norm.h5ad'))

adata = adata[adata.obs.Subclass.isin(['Oligodendrocyte'])].copy()
adata.var.index = adata.var.feature_name
adata.var.index.name = 'gene_id'
adata.raw = adata

sc.pp.filter_genes(adata, min_cells=5)
adata.write(os.path.join(output_dir, 'MTG_oligo_sciMultiLaMA_norm.h5ad'))
metadata = adata.obs.copy()
metadata.to_csv(os.path.join(output_dir, 'MTG_oligo_metadata.csv'))

cell_embedding_df = pd.read_csv(
    os.path.join(emb_path, 'Oligodendrocyte_MTG_sciMultiLaMA_CELL_embedding.csv'),
    index_col=0
)

cell_embedding_df = cell_embedding_df.loc[adata.obs.index]
adata.obsm['X_sciMultiLaMA'] = cell_embedding_df.values

sc.pp.neighbors(adata, n_neighbors=25, use_rep='X_sciMultiLaMA')
sc.tl.umap(adata)
adata.write(os.path.join(output_dir, 'MTG_oligo_sciMultiLaMA.h5ad'))
