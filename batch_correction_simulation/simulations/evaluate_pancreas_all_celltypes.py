"""evaluate_pancreas.py on the all-cell-type pancreas bags from simulate_pancreas_all_celltypes.py.

Same model, seeds (5-14), no_batch / with_batch conditions and metrics. This cohort has 5 bags
per (batch, label) pair, and TEST_PER_STATUS_PER_BATCH = 2 of them form the fixed test set.
Unlike evaluate_pancreas.py, the no_batch UMAP is coloured by batch too.

Reads $DATA_DIR/pancreatic_w_batch/pancreas_synthetic_baron_muraro_all_celltypes.h5ad. Writes
the same files as evaluate_pancreas.py (metric tables, per-run copies, test_sample_ids.csv,
models_<cond>_seed_<s>/, first-seed umap_<cond>_{batch,label,celltype}.png) to
$RESULTS_DIR/R1-q1/Results_pancreas_synth_cellect_all_celltypes/. Uses a GPU when available.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)

import cellection

DATA_PATH = (
    f"{DATA_DIR}/pancreatic_w_batch/"
    "pancreas_synthetic_baron_muraro_all_celltypes.h5ad"
)
SAVE_DIR = (
    f"{RESULTS_DIR}/R1-q1/"
    "Results_pancreas_synth_cellect_all_celltypes"
)
os.makedirs(SAVE_DIR, exist_ok=True)

batch_key = "batch"
sample_key = "sample_id"
task_key = "b_status"
cell_type_key = "celltype"
TEST_PER_STATUS_PER_BATCH = 2  # bags per (batch, label) pair held out for test
TEST_SPLIT_SEED = 0  # seed of the test-bag selection, the same for every training seed
DECISION_THRESHOLD = 0.5  # with_b probability at or above which a bag is called with_b


def compute_binary_metrics(true_df, pred_df, pos_label):
    """AUROC, AUPRC, macro-F1, per-class F1 and confusion matrix, with `pos_label` as positive.

    true_df is one-hot and pred_df holds class probabilities, same columns; a bag is called
    positive at probability >= DECISION_THRESHOLD.
    """
    label_cols = list(true_df.columns)
    if pos_label not in label_cols:
        raise ValueError(f"pos_label {pos_label} not in {label_cols}")
    pos_idx = label_cols.index(pos_label)
    y_true = np.argmax(true_df.values, axis=1)
    y_true_bin = (y_true == pos_idx).astype(int)
    y_prob = pred_df.values[:, pos_idx]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    return {
        "AUROC": roc_auc_score(y_true_bin, y_prob),
        "AUPRC": average_precision_score(y_true_bin, y_prob),
        "MacroF1": f1_score(y_true_bin, y_pred, average="macro"),
        "F1_pos": f1_score(y_true_bin, y_pred, pos_label=1),
        "F1_neg": f1_score(y_true_bin, y_pred, pos_label=0),
        "conf_mat": confusion_matrix(y_true_bin, y_pred),
    }


def fix_bag_orientation(dataset, enc_in, pn_in, tag, prefer_enc=True):
    """Transpose the bags in dataset.Xs to (n_cells, n_features) if they are the other way round.

    enc_in and pn_in are the instance-encoder and PointNet input widths. Only the first bag is
    checked and all bags are flipped together; ValueError if neither width matches.
    """
    if not dataset.Xs:
        return
    bag = dataset.Xs[0]
    valid_dims = [d for d in [enc_in, pn_in] if d is not None]
    if prefer_enc and enc_in is not None:
        if bag.shape[1] == enc_in:
            return
        if bag.shape[0] == enc_in:
            dataset.Xs = [b.T for b in dataset.Xs]
            return
    if pn_in is not None:
        if bag.shape[1] == pn_in:
            return
        if bag.shape[0] == pn_in:
            dataset.Xs = [b.T for b in dataset.Xs]
            return
    raise ValueError(f"[{tag}] Bag shape {bag.shape} does not match expected feature dims {valid_dims}.")


def get_expected_features(obj):
    """Input widths of the instance encoder (None if absent) and of the PointNet backbone."""
    enc_in = None
    if obj.InstanceEncoder:
        for layer in obj.enc.hidden_rep:
            if isinstance(layer, nn.Linear):
                enc_in = layer.in_features
                break
    pn_in = obj.pn.backbone.shared_mlp1.conv1.conv1d.in_channels
    return enc_in, pn_in


def compute_cell_embeddings(obj, adata, chunk_size=2048):
    """Instance-encoder embedding of every cell in `adata`, in adata order.

    Rebuilds the training input: expression plus, with batch conditioning, the one-hot batch columns
    in training order (obj.batch_dummy_columns). Runs on CPU, chunk_size cells at a time.
    """
    if not obj.InstanceEncoder or obj.enc_checkpoint is None:
        raise ValueError("Instance encoder is required to compute cell embeddings.")
    obj.enc_checkpoint.to("cpu")
    obj.enc_checkpoint.eval()
    X = adata.X
    batch_dummy = None
    if obj.batch_key is not None:
        batch_dummy = pd.get_dummies(adata.obs[obj.batch_key].copy()).astype(int)
        if hasattr(obj, "batch_dummy_columns"):
            batch_dummy = batch_dummy.reindex(columns=obj.batch_dummy_columns, fill_value=0)
        else:
            obj.batch_dummy_columns = batch_dummy.columns.tolist()
        batch_dummy = batch_dummy.values
    embeddings = []
    with torch.no_grad():
        for start in range(0, X.shape[0], chunk_size):
            end = min(start + chunk_size, X.shape[0])
            if sparse.issparse(X):
                x_chunk = X[start:end].toarray()
            else:
                x_chunk = X[start:end]
            if batch_dummy is not None:
                x_chunk = np.hstack([x_chunk, batch_dummy[start:end]])
            x_chunk = torch.tensor(x_chunk, dtype=torch.float32)
            emb = obj.enc_checkpoint(x_chunk, None)
            embeddings.append(emb.detach().cpu().numpy())
    return np.vstack(embeddings)


def stratified_test_split(adata, per_status=2):
    """`per_status` random bag IDs per (batch, label) pair (all of them if the pair has no more).

    The RNG is seeded with TEST_SPLIT_SEED here, so every training seed gets the same test bags.
    """
    test_ids = []
    rng = np.random.default_rng(TEST_SPLIT_SEED)
    for batch in adata.obs[batch_key].unique():
        sub = adata[adata.obs[batch_key] == batch]
        for status in adata.obs[task_key].unique():
            ids = sub.obs[sub.obs[task_key] == status][sample_key].unique().tolist()
            if len(ids) <= per_status:
                test_ids.extend(ids)
            else:
                test_ids.extend(rng.choice(ids, size=per_status, replace=False).tolist())
    return list(dict.fromkeys(test_ids))


def run_cellection(
    adata_train,
    adata_test,
    adata_full,
    batch_setting,
    tag,
    seed,
    batch_categories=None,
    plot_umap=False,
):
    """Train on adata_train; return (bag metrics on adata_test, cell metrics on adata_full).

    batch_setting is None ("no_batch") or batch_key ("with_batch"). UMAPs are saved if plot_umap.
    """
    # Fixed batch categories so pd.get_dummies yields the same one-hot columns for every subset.
    if batch_setting is not None and batch_categories is not None:
        adata_train.obs[batch_key] = pd.Categorical(adata_train.obs[batch_key], categories=batch_categories)
        adata_test.obs[batch_key] = pd.Categorical(adata_test.obs[batch_key], categories=batch_categories)
        adata_full.obs[batch_key] = pd.Categorical(adata_full.obs[batch_key], categories=batch_categories)
    obj = cellection.cellectiion_object(
        adata=adata_train,
        task_type="classification",
        task_key=task_key,
        sample_key=sample_key,
        batch_key=batch_setting,
        input_type="measurement",
        sparse_input=True,
        InstanceEncoder=True,
        val_size=0.2,
        aggregator="gated_attention",
        global_features=256,
        attention_dim=64,
        max_epochs=200,
        learning_rate=1e-4,
        batch_size=15,
        seed=seed,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hidden_layer=[256, 32],
        activation=nn.ReLU(),
        layernorm=True,
        dropout_rate=0.1,
        save_model=True,
        save_path=f"{SAVE_DIR}/models_{tag}_seed_{seed}",
    )

    obj.prepare()
    enc_in, pn_in = get_expected_features(obj)
    fix_bag_orientation(obj.mil_dataset, enc_in, pn_in, f"{tag}-train", prefer_enc=obj.InstanceEncoder)
    obj.train()

    # Training one-hot column order, reused by compute_cell_embeddings().
    if batch_setting is not None:
        obj.batch_dummy_columns = pd.get_dummies(adata_train.obs[batch_key].copy()).astype(int).columns.tolist()

    # inference runs on whatever obj.adata holds
    obj.adata = adata_test.copy()
    obj.prepare_inference()
    enc_in, pn_in = get_expected_features(obj)
    fix_bag_orientation(obj.mil_dataset, enc_in, pn_in, f"{tag}-inference", prefer_enc=obj.InstanceEncoder)
    _, true_df, pred_df, _, _, _ = obj.inference()

    metrics = compute_binary_metrics(true_df, pred_df, pos_label="with_b")
    metrics["seed"] = seed
    metrics["condition"] = tag

    # cell-level metrics on all cells, training bags included
    emb_key = f"X_cel_pancreas_all_{tag}"
    adata_full.obsm[emb_key] = compute_cell_embeddings(obj, adata_full)
    sc.pp.neighbors(adata_full, use_rep=emb_key)
    sc.tl.leiden(adata_full, key_added=f"leiden_{tag}")
    ari = adjusted_rand_score(adata_full.obs[cell_type_key], adata_full.obs[f"leiden_{tag}"])
    nmi = normalized_mutual_info_score(adata_full.obs[cell_type_key], adata_full.obs[f"leiden_{tag}"])
    embedding_metrics = {
        "seed": seed,
        "condition": tag,
        "ARI": ari,
        "NMI": nmi,
    }
    try:
        from scib.metrics import silhouette_batch, silhouette
        embedding_metrics["batchASW"] = silhouette_batch(
            adata_full, batch_key=batch_key, group_key=cell_type_key, embedding_key=emb_key
        )
        embedding_metrics["celltypeASW"] = silhouette(adata_full, group_key=cell_type_key, embed=emb_key)
    except Exception:
        # scib missing or failing: sklearn silhouettes. This batchASW is 1 - silhouette(batch
        # labels) over all cells, not scib's per-cell-type silhouette_batch; don't mix the two.
        X_emb = adata_full.obsm[emb_key]
        if sparse.issparse(X_emb):
            X_emb = X_emb.toarray()
        batch_labels = adata_full.obs[batch_key].values
        cell_labels = adata_full.obs[cell_type_key].values
        embedding_metrics["batchASW"] = np.nan
        embedding_metrics["celltypeASW"] = np.nan
        if len(np.unique(batch_labels)) >= 2:
            embedding_metrics["batchASW"] = 1.0 - silhouette_score(X_emb, batch_labels)
        if len(np.unique(cell_labels)) >= 2:
            embedding_metrics["celltypeASW"] = silhouette_score(X_emb, cell_labels)

    if plot_umap:
        sc.pp.neighbors(adata_full, use_rep=emb_key)
        sc.tl.umap(adata_full)
        sc.pl.umap(adata_full, color=batch_key, show=False)
        plt.savefig(f"{SAVE_DIR}/umap_{tag}_batch.png", dpi=150, bbox_inches="tight")
        plt.close()
        sc.pl.umap(adata_full, color=task_key, show=False)
        plt.savefig(f"{SAVE_DIR}/umap_{tag}_label.png", dpi=150, bbox_inches="tight")
        plt.close()
        sc.pl.umap(adata_full, color=cell_type_key, show=False)
        plt.savefig(f"{SAVE_DIR}/umap_{tag}_celltype.png", dpi=150, bbox_inches="tight")
        plt.close()

    return metrics, embedding_metrics


def main():
    adata = sc.read_h5ad(DATA_PATH)
    adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    adata.obs[task_key] = adata.obs[task_key].astype("category")
    batch_categories = adata.obs[batch_key].cat.categories

    test_ids = stratified_test_split(adata, per_status=TEST_PER_STATUS_PER_BATCH)
    adata_test = adata[adata.obs[sample_key].isin(test_ids)].copy()
    adata_train = adata[~adata.obs[sample_key].isin(test_ids)].copy()
    pd.Series(test_ids).to_csv(f"{SAVE_DIR}/test_sample_ids.csv", index=False, header=False)

    SEEDS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    # with_batch is CELLECTION's batch conditioning: the one-hot batch is appended to every
    # cell's expression vector at the instance encoder.
    conditions = [("no_batch", None), ("with_batch", batch_key)]
    clf_rows = []
    emb_rows = []
    for seed in SEEDS:
        for tag, batch_setting in conditions:
            plot_umap = seed == SEEDS[0]
            clf_metrics, emb_metrics = run_cellection(
                adata_train.copy(),
                adata_test.copy(),
                adata.copy(),
                batch_setting,
                tag,
                seed,
                batch_categories=batch_categories,
                plot_umap=plot_umap,
            )
            clf_rows.append(clf_metrics)
            emb_rows.append(emb_metrics)
            pd.DataFrame([clf_metrics]).to_csv(
                f"{SAVE_DIR}/classification_metrics_{tag}_seed_{seed}.csv", index=False
            )
            pd.DataFrame([emb_metrics]).to_csv(
                f"{SAVE_DIR}/embedding_metrics_{tag}_seed_{seed}.csv", index=False
            )

    pd.DataFrame(clf_rows).to_csv(f"{SAVE_DIR}/classification_metrics.csv", index=False)
    pd.DataFrame(emb_rows).to_csv(f"{SAVE_DIR}/embedding_metrics.csv", index=False)


if __name__ == "__main__":
    main()
