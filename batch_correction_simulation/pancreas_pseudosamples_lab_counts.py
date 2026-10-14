"""pancreas_pseudosamples.py plus cell-count tables per study (lab), written to Results_pancreas_new/.

The pseudo-sample split, model, seeds (0-4), conditions and metrics are the same. Added: cells
per study and per study x cell type, before and after the split (the split drops samples
lacking beta or non-beta cells), and for each study a cell type x (with_b, no_b) table of
counts and column percentages, to compare the composition of the two pseudo-sample types
within a lab. The seed-0 UMAPs are also coloured by cell type.

Reads $DATA_DIR/pancreatic_w_batch/Fig2_obj.h5ad. Writes to
$RESULTS_DIR/R1-q1/Results_pancreas_new/:
    classification_metrics.csv, embedding_metrics.csv (one row per seed x condition)
    classification_metrics_<cond>_seed_<s>.csv, embedding_metrics_<cond>_seed_<s>.csv
    models_<cond>_seed_<s>/
    umap_<cond>_colored_by_{batch,target,celltype}.png (seed 0)
    num_cells_per_lab.csv, num_cells_per_lab_by_celltype.csv (before the split)
    num_cells_after_pseudo_per_lab.csv, num_cells_after_pseudo_by_lab_celltype.csv
    <lab>_celltype_with_vs_without_beta_{COUNTS,PERCENT}.csv
Uses a GPU when available.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

import cellection


DATA_PATH = f"{DATA_DIR}/pancreatic_w_batch/Fig2_obj.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R1-q1/Results_pancreas_new"
os.makedirs(SAVE_DIR, exist_ok=True)

batch_key = "dataset"
cell_type_key = "celltype"
b_cell_label = "beta"
DECISION_THRESHOLD = 0.5  # with_b probability at or above which a bag is called with_b


def make_pseudo_labels(adata, cell_type_key, b_cell_label, seed=0):
    """Split each sample into a with-beta and a no-beta pseudo-sample.

    The with-beta half gets all b_cell_label cells plus k = (n_nonbeta - n_beta) // 2 non-beta cells
    (none if n_nonbeta <= n_beta). Returns (kept cells with obs sample_id, pseudo_sample and
    pseudo_label in {"with_b", "no_b"}; list of dropped sample IDs).
    """
    obs = adata.obs.copy()
    # sample ID = barcode prefix before the first '.'
    obs["sample_id"] = obs.index.to_series().str.split(".", n=1).str[0]
    rng = np.random.default_rng(seed)
    pseudo_sample = pd.Series(index=obs.index, dtype="object")
    pseudo_label = pd.Series(index=obs.index, dtype="object")
    dropped = []

    for sample_id, idx in obs.groupby("sample_id").groups.items():
        sub = obs.loc[idx]
        b_mask = sub[cell_type_key] == b_cell_label
        n_b = int(b_mask.sum())
        n_nonb = int((~b_mask).sum())
        if n_b == 0 or n_nonb == 0:
            dropped.append(sample_id)
            continue

        b_idx = sub.index[b_mask].to_numpy()
        nonb_idx = sub.index[~b_mask].to_numpy()

        # k = (n_nonb - n_b) // 2 evens out the halves (n_b + k ~= n_nonb - k). Taking k / n_nonb
        # of each non-beta type keeps the non-beta composition the same in both halves; rounding
        # is then fixed by trimming or topping up at random to exactly k.
        if n_nonb <= n_b:
            nonb_for_with = np.array([], dtype=object)
            nonb_for_no = nonb_idx
        else:
            k = (n_nonb - n_b) // 2
            ratio = k / n_nonb
            nonb_for_with = []
            nonb_for_no = []
            for _, ct_idx in sub.loc[nonb_idx].groupby(cell_type_key).groups.items():
                ct_idx = np.array(ct_idx)
                rng.shuffle(ct_idx)
                ct_k = int(round(len(ct_idx) * ratio))
                nonb_for_with.extend(ct_idx[:ct_k])
                nonb_for_no.extend(ct_idx[ct_k:])
            nonb_for_with = list(nonb_for_with)
            nonb_for_no = list(nonb_for_no)
            if len(nonb_for_with) > k:
                extra = rng.choice(nonb_for_with, size=len(nonb_for_with) - k, replace=False)
                extra = set(extra.tolist())
                nonb_for_with = [i for i in nonb_for_with if i not in extra]
                nonb_for_no.extend(list(extra))
            elif len(nonb_for_with) < k:
                needed = k - len(nonb_for_with)
                if needed > 0 and len(nonb_for_no) >= needed:
                    add = rng.choice(nonb_for_no, size=needed, replace=False)
                    add = set(add.tolist())
                    nonb_for_no = [i for i in nonb_for_no if i not in add]
                    nonb_for_with.extend(list(add))
            nonb_for_with = np.array(nonb_for_with)
            nonb_for_no = np.array(nonb_for_no)

        with_b = np.concatenate([b_idx, nonb_for_with])
        no_b = nonb_for_no
        pseudo_sample.loc[with_b] = f"{sample_id}_with_{b_cell_label}"
        pseudo_label.loc[with_b] = "with_b"
        pseudo_sample.loc[no_b] = f"{sample_id}_no_{b_cell_label}"
        pseudo_label.loc[no_b] = "no_b"

    obs["pseudo_sample"] = pseudo_sample
    obs["pseudo_label"] = pseudo_label
    adata_pseudo = adata[obs["pseudo_sample"].notna()].copy()
    adata_pseudo.obs = obs.loc[adata_pseudo.obs.index].copy()
    return adata_pseudo, dropped


def get_val_sample_ids(obj, adata, sample_key):
    """IDs of the bags CELLECTION held out for early stopping.

    Each obj.val_dataset entry carries the bag's cell IDs as its 4th field; they are mapped back to
    `sample_key` through adata.obs.
    """
    val_sample_ids = set()
    for (_, _, _, meta_id) in obj.val_dataset:
        if isinstance(meta_id, (list, np.ndarray)):
            cell_ids = list(meta_id)
        else:
            cell_ids = [meta_id]
        val_sample_ids.update(
            adata.obs.loc[cell_ids, sample_key].unique().tolist()
        )
    return val_sample_ids


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
    raise ValueError(
        f"[{tag}] Bag shape {bag.shape} does not match expected "
        f"feature dims {valid_dims}."
    )


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
        batch_dummy = pd.get_dummies(
            adata.obs[obj.batch_key].copy()
        ).astype(int)
        if hasattr(obj, "batch_dummy_columns"):
            batch_dummy = batch_dummy.reindex(
                columns=obj.batch_dummy_columns,
                fill_value=0
            )
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


def run_cellection(adata, batch_setting, tag, seed, batch_categories=None, plot_umap=False):
    """Train one model on all pseudo-samples; return (bag-level metrics, cell-level metrics).

    batch_setting is None ("no_batch") or batch_key ("with_batch"). Bag metrics are computed on
    CELLECTION's early-stopping validation bags. If scib fails, ASW falls back to sklearn
    silhouettes. Uses the module-level sample_key and task_key.
    """
    # Fixed batch categories so pd.get_dummies yields the same one-hot columns for every subset.
    if batch_setting is not None and batch_categories is not None:
        adata.obs[batch_key] = pd.Categorical(
            adata.obs[batch_key],
            categories=batch_categories
        )
    obj = cellection.cellectiion_object(
        adata=adata,
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
        max_epochs=50,
        learning_rate=1e-4,
        batch_size=20,
        seed=seed,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hidden_layer=[256, 32],
        activation=nn.ReLU(),
        layernorm=True,
        dropout_rate=0.1,
        save_model=True,
        save_path=f"{SAVE_DIR}/models_{tag}_seed_{seed}",
    )

    print(f"\nRunning CELLECTION ({tag}) ...")
    obj.prepare()
    enc_in, pn_in = get_expected_features(obj)
    if obj.mil_dataset.Xs:
        print(
            f"[DEBUG {tag}-train] enc_in={enc_in} pn_in={pn_in} "
            f"bag0={obj.mil_dataset.Xs[0].shape}"
        )
    fix_bag_orientation(
        obj.mil_dataset,
        enc_in,
        pn_in,
        f"{tag}-train",
        prefer_enc=obj.InstanceEncoder
    )
    obj.train()

    # Training one-hot column order, reused by compute_cell_embeddings().
    if batch_setting is not None:
        obj.batch_dummy_columns = pd.get_dummies(
            adata.obs[batch_key].copy()
        ).astype(int).columns.tolist()

    emb_key = f"X_cel_pseudo_{tag}"
    adata.obsm[emb_key] = compute_cell_embeddings(obj, adata)

    # drop any stored UMAP so this condition's is computed from scratch
    if "X_umap" in adata.obsm:
        del adata.obsm["X_umap"]

    sc.pp.neighbors(adata, use_rep=emb_key)
    sc.tl.umap(adata)
    if plot_umap:
        sc.pl.umap(
            adata,
            color=batch_key,
            title=f"CELLECTION ({tag}) – colored by batch",
            show=False,
        )
        plt.savefig(
            f"{SAVE_DIR}/umap_{tag}_colored_by_batch.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        sc.pl.umap(
            adata,
            color=task_key,
            title=f"CELLECTION ({tag}) – colored by target",
            show=False,
        )
        plt.savefig(
            f"{SAVE_DIR}/umap_{tag}_colored_by_target.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        sc.pl.umap(
            adata,
            color=cell_type_key,
            title=f"CELLECTION ({tag}) – colored by cell type",
            show=False,
        )
        plt.savefig(
            f"{SAVE_DIR}/umap_{tag}_colored_by_celltype.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # No separate test set: score the validation bags, which also drove early stopping.
    val_sample_ids = get_val_sample_ids(obj, adata, sample_key)
    adata_val = adata[adata.obs[sample_key].isin(val_sample_ids)].copy()
    if batch_setting is not None and batch_categories is not None:
        adata_val.obs[batch_key] = pd.Categorical(
            adata_val.obs[batch_key],
            categories=batch_categories
        )
    for col in ["attention_score_raw", "attention_score_norm_cellnum"]:
        if col in adata_val.obs.columns:
            adata_val.obs = adata_val.obs.drop(columns=[col])
    obj.adata = adata_val
    obj.prepare_inference()
    enc_in, pn_in = get_expected_features(obj)
    if obj.mil_dataset.Xs:
        print(
            f"[DEBUG {tag}-inference] enc_in={enc_in} pn_in={pn_in} "
            f"bag0={obj.mil_dataset.Xs[0].shape}"
        )
    fix_bag_orientation(
        obj.mil_dataset,
        enc_in,
        pn_in,
        f"{tag}-inference",
        prefer_enc=obj.InstanceEncoder
    )
    _, true_df, pred_df, _, _, _ = obj.inference()

    metrics = compute_binary_metrics(true_df, pred_df, pos_label="with_b")
    metrics["seed"] = seed
    metrics["condition"] = tag

    sc.pp.neighbors(adata, use_rep=emb_key)
    sc.tl.leiden(adata, key_added=f"leiden_{tag}")
    ari = adjusted_rand_score(adata.obs[cell_type_key], adata.obs[f"leiden_{tag}"])
    nmi = normalized_mutual_info_score(adata.obs[cell_type_key], adata.obs[f"leiden_{tag}"])
    embedding_metrics = {
        "seed": seed,
        "condition": tag,
        "ARI": ari,
        "NMI": nmi,
    }

    try:
        from scib.metrics import silhouette_batch, silhouette
        batch_asw = silhouette_batch(
            adata,
            batch_key=batch_key,
            group_key=cell_type_key,
            embedding_key=emb_key,
        )
        cell_asw = silhouette(adata, group_key=cell_type_key, embed=emb_key)
        embedding_metrics["batchASW"] = batch_asw
        embedding_metrics["celltypeASW"] = cell_asw
    except Exception as exc:
        print(f"[WARN] scib metrics not available: {exc}")
        # scib missing or failing: sklearn silhouettes. This batchASW is 1 - silhouette(batch
        # labels) over all cells, not scib's per-cell-type silhouette_batch; don't mix the two.
        X_emb = adata.obsm[emb_key]
        if sparse.issparse(X_emb):
            X_emb = X_emb.toarray()
        batch_labels = adata.obs[batch_key].values
        cell_labels = adata.obs[cell_type_key].values
        if len(np.unique(batch_labels)) < 2:
            embedding_metrics["batchASW"] = np.nan
        else:
            embedding_metrics["batchASW"] = 1.0 - silhouette_score(X_emb, batch_labels)
        if len(np.unique(cell_labels)) < 2:
            embedding_metrics["celltypeASW"] = np.nan
        else:
            embedding_metrics["celltypeASW"] = silhouette_score(X_emb, cell_labels)

    return metrics, embedding_metrics


adata = sc.read_h5ad(DATA_PATH)
print("Loaded:", adata)
print("Batch categories:", adata.obs[batch_key].unique())
print("Cell types:", adata.obs[cell_type_key].unique())

# Composition before the pseudo-sample split
cells_per_lab = adata.obs[batch_key].value_counts().sort_index()
print("\nNumber of cells per lab:")
print(cells_per_lab)
cells_per_lab.to_csv(f"{SAVE_DIR}/num_cells_per_lab.csv")

cells_per_lab_celltype = (
    adata.obs
    .groupby([batch_key, cell_type_key])
    .size()
    .unstack(fill_value=0)
)

print("\nNumber of cells per lab × cell type:")
print(cells_per_lab_celltype)
cells_per_lab_celltype.to_csv(
    f"{SAVE_DIR}/num_cells_per_lab_by_celltype.csv"
)
batch_categories = adata.obs[batch_key].astype("category").cat.categories

adata, dropped_samples = make_pseudo_labels(adata, cell_type_key, b_cell_label, seed=0)

# Per-lab tables after the split: rows = cell types, columns = with_b / no_b pseudo-samples
labs = adata.obs[batch_key].unique()

for lab in labs:
    ad_lab = adata[adata.obs[batch_key] == lab].copy()

    count_table = (
        ad_lab.obs
        .groupby([cell_type_key, "pseudo_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["with_b", "no_b"], fill_value=0)
        .sort_index()
    )

    print(f"\n[LAB = {lab}] Cell counts")
    print(count_table)

    count_table.to_csv(
        f"{SAVE_DIR}/{lab}_celltype_with_vs_without_beta_COUNTS.csv"
    )

    # Column percentages (each pseudo-label column sums to 100)
    percent_table = count_table.div(count_table.sum(axis=0), axis=1) * 100

    print(f"\n[LAB = {lab}] Cell percentages")
    print(percent_table.round(2))

    percent_table.to_csv(
        f"{SAVE_DIR}/{lab}_celltype_with_vs_without_beta_PERCENT.csv"
    )

# Composition after the split, which drops samples lacking beta or non-beta cells
cells_after_pseudo_per_lab = (
    adata.obs[batch_key]
    .value_counts()
    .sort_index()
)
print("\nNumber of cells per lab AFTER pseudo-labeling:")
print(cells_after_pseudo_per_lab)
cells_after_pseudo_per_lab.to_csv(
    f"{SAVE_DIR}/num_cells_after_pseudo_per_lab.csv"
)
cells_after_pseudo = (
    adata.obs
    .groupby([batch_key, cell_type_key])
    .size()
    .unstack(fill_value=0)
)

print("\nCells AFTER pseudo-label filtering (used for MIL bags):")
print(cells_after_pseudo)

cells_after_pseudo.to_csv(
    f"{SAVE_DIR}/num_cells_after_pseudo_by_lab_celltype.csv"
)
if dropped_samples:
    print(f"Dropped {len(dropped_samples)} samples without both B and non-B cells.")
adata.obs[batch_key] = pd.Categorical(adata.obs[batch_key], categories=batch_categories)

# sample_key / task_key are module-level globals read inside run_cellection().
sample_key = "pseudo_sample"
task_key = "pseudo_label"
adata.obs[task_key] = adata.obs[task_key].astype("category")

SEEDS = [0,1,2,3,4]
# with_batch is CELLECTION's batch conditioning: the one-hot batch is appended to every
# cell's expression vector at the instance encoder.
conditions = [("no_batch", None), ("with_batch", batch_key)]
clf_rows = []
emb_rows = []
for seed in SEEDS:
    for tag, batch_setting in conditions:
        plot_umap = seed == SEEDS[0]
        adata_run = adata.copy()
        clf_metrics, emb_metrics = run_cellection(
            adata_run,
            batch_setting,
            tag,
            seed,
            batch_categories=batch_categories,
            plot_umap=plot_umap,
        )
        clf_rows.append(clf_metrics)
        emb_rows.append(emb_metrics)
        pd.DataFrame([clf_metrics]).to_csv(
            f"{SAVE_DIR}/classification_metrics_{tag}_seed_{seed}.csv",
            index=False
        )
        pd.DataFrame([emb_metrics]).to_csv(
            f"{SAVE_DIR}/embedding_metrics_{tag}_seed_{seed}.csv",
            index=False
        )

pd.DataFrame(clf_rows).to_csv(f"{SAVE_DIR}/classification_metrics.csv", index=False)
pd.DataFrame(emb_rows).to_csv(f"{SAVE_DIR}/embedding_metrics.csv", index=False)
