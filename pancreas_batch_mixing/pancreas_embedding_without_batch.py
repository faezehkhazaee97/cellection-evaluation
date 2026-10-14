"""No-covariate arm of pancreas_embedding_with_batch.py: USE_BATCH = False, no study columns in the input.

Everything else matches, except that this run uses seed 0 where the with-batch run uses seed 1. Studies
are still the bags and still the labels for ASW_batch.

Writes $RESULTS_DIR/R1-q4b/Results/pancreas_batch_eval_no_batch.csv (no "celllevel" in the name, so
pancreas_attention_with_batch.py run with USE_BATCH = False overwrites the same file) and checkpoints to
tmp_pancreas_batch_nobatch/train_val_classification_SEED0/ in the same folder.
"""

# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import numpy as np
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

USE_BATCH = False  # no batch covariate in the input
BATCH_KEY = "dataset"  # technical batch (study); also the bag
CELLTYPE_KEY = "celltype"   # used only for evaluation
DUMMY_LABEL = "train_val"   # placeholder training label


adata = sc.read_h5ad(DATA_PATH)

print(adata)
if USE_BATCH:
    print("Batch counts:\n", adata.obs[BATCH_KEY].value_counts())
else:
    print("No batch covariate used.")
print("Cell types:\n", adata.obs[CELLTYPE_KEY].value_counts())


def build_input_matrix(adata, batch_key):
    """Cells x features encoder input, built as cellection's prepare() does: dense expression, then
    one-hot batch columns (pd.get_dummies order) when batch_key is not None.
    """
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X_df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    if batch_key is not None:
        batch_dummy = pd.get_dummies(adata.obs[batch_key]).astype(int)
        X_df = pd.concat([X_df, batch_dummy], axis=1)
    return X_df


def get_cell_embeddings(adata, batch_key, enc_model, device, batch_size=1024):
    """Instance-encoder embedding of every cell, (n_cells, hidden_layer[-1]) in adata.obs order.

    The full input matrix is moved to `device` at once; only the forward pass is chunked by batch_size.
    """
    enc_model.eval()
    X_df = build_input_matrix(adata, batch_key)
    X = torch.tensor(X_df.values, dtype=torch.float32, device=device)
    outputs = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = X[start:start + batch_size]
            emb = enc_model(batch, None)
            outputs.append(emb.cpu().numpy())
    return np.vstack(outputs)


obj = cellection.cellectiion_object(
    adata=adata,
    task_type="classification",
    task_key=DUMMY_LABEL,
    sample_key = "dataset",  # one bag per study
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
    seed=0,
    save_model=True,
    save_path=f"{SAVE_DIR}/tmp_pancreas_batch_nobatch",
)

obj.prepare()
obj.train()

enc_model = obj.enc_checkpoint if obj.InstanceEncoder else None
if enc_model is None:
    raise ValueError("InstanceEncoder is required for cell-level embeddings.")
enc_model = enc_model.to(DEVICE)

cell_emb = get_cell_embeddings(
    adata,
    BATCH_KEY if USE_BATCH else None,
    enc_model,
    DEVICE,
    batch_size=1024,
)
adata.obsm["X_cellection"] = cell_emb

print("Cell embedding shape:", adata.obsm["X_cellection"].shape)


X = adata.obsm["X_cellection"]
celltype_labels = adata.obs[CELLTYPE_KEY].values

batch_labels = adata.obs[BATCH_KEY].values
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
