"""Does a study-level batch covariate change how CELLECTION's attention scores mix across studies?

The data are human pancreas cells from several studies; obs `dataset` (the study) is both the batch and
the bag. The obs column `train_val` is only a placeholder label: the model is trained (seed 1) for its
per-cell outputs, not for the prediction. USE_BATCH = True appends the one-hot study to the expression
input; set it to False for the no-batch arm. After training, inference runs over every cell and the
per-cell attention table (raw score, and score times bag size) is scored with
    ASW_batch     silhouette by `dataset`, raw (lower = better mixing; not the 1 - ASW convention)
    ASW_celltype  silhouette by `celltype`
    ARI, NMI      KMeans (k = number of cell types, random_state=0) against `celltype`

Reads $DATA_DIR/pancreatic_w_batch/Fig2_obj.h5ad. Writes one row (setting, ASW_batch, ASW_celltype, ARI,
NMI) to $RESULTS_DIR/R1-q4b/Results/pancreas_batch_eval_{with_batch|no_batch}.csv and checkpoints to
tmp_pancreas_batch_baglevel/train_val_classification_SEED1/ in the same folder.
"""

# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import pandas as pd
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import cellection

DATA_PATH = f"{DATA_DIR}/pancreatic_w_batch/Fig2_obj.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R1-q4b/Results"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_KEY = "dataset"       # technical batch (study); also the bag
CELLTYPE_KEY = "celltype"   # used only for evaluation
DUMMY_LABEL = "train_val"   # placeholder training label
USE_BATCH = True            # append the one-hot batch covariate to the input

adata = sc.read_h5ad(DATA_PATH)
print("obs columns:")
print(adata.obs.columns.tolist())
print(adata)
print("Batch counts:\n", adata.obs[BATCH_KEY].value_counts())
print("Cell types:\n", adata.obs[CELLTYPE_KEY].value_counts())

obj = cellection.cellectiion_object(
    adata=adata,
    task_type="classification",
    task_key=DUMMY_LABEL,
    sample_key=BATCH_KEY,                 # one bag per study
    batch_key=BATCH_KEY if USE_BATCH else None,
    model_type="classification",
    input_type="measurement",
    sparse_input=True,
    InstanceEncoder=True,
    val_size=0.2,
    aggregator="gated_attention",
    global_features=256,
    attention_dim=64,
    hidden_layer=[256, 64],
    activation=torch.nn.ReLU(),
    dropout_rate=0.1,
    learning_rate=1e-4,
    batch_size=32,
    max_epochs=200,
    patience=20,
    device=DEVICE,
    seed=1,
    save_model=True,
    save_path=f"{SAVE_DIR}/tmp_pancreas_batch_baglevel",
)

obj.prepare()
obj.train()

obj.prepare_inference()

(
    sample_metadata,
    true_label_df,
    pred_label_df,
    embedding_df,
    global_feature_df,
    attention_mtx_raw_df,
) = obj.inference()

print("Embedding shape:", embedding_df.shape)
print("Embedding index matches adata:", embedding_df.index.equals(adata.obs.index))

# embedding_df has one row per bag (study), so the check above prints False; it is not used.
# attention_mtx_raw_df has one row per cell in the order inference visits the bags, and .values below
# is taken without reindexing, so the index check printed next has to be True.

print(attention_mtx_raw_df.shape)
print(attention_mtx_raw_df.index.equals(adata.obs.index))

# Here X_cellection is the two attention columns; the embedding scripts put encoder embeddings there.
adata.obsm["X_cellection"] = attention_mtx_raw_df.values

assert adata.obsm["X_cellection"].shape[0] == adata.n_obs


X = adata.obsm["X_cellection"]
batch_labels = adata.obs[BATCH_KEY].values
celltype_labels = adata.obs[CELLTYPE_KEY].values

# Raw silhouette by study: lower is better mixing (no 1 - ASW rescaling).
asw_batch = silhouette_score(X, batch_labels)
asw_celltype = silhouette_score(X, celltype_labels)

k = adata.obs[CELLTYPE_KEY].nunique()
clusters = KMeans(n_clusters=k, random_state=0).fit_predict(X)

ari = adjusted_rand_score(celltype_labels, clusters)
nmi = normalized_mutual_info_score(celltype_labels, clusters)

results = pd.DataFrame([{
    "setting": "with_batch" if USE_BATCH else "no_batch",
    "ASW_batch": asw_batch,
    "ASW_celltype": asw_celltype,
    "ARI": ari,
    "NMI": nmi,
}])

out_path = f"{SAVE_DIR}/pancreas_batch_eval_{'with_batch' if USE_BATCH else 'no_batch'}.csv"
results.to_csv(out_path, index=False)

print("\n===== PANCREAS BATCH EVALUATION =====")
print(results)
print(f"Saved to: {out_path}")