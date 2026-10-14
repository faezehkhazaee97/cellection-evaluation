"""Lupus SLE vs healthy: per-bin CELLECTION performance on all cells, with donors in 4 cell-count bins.

Bins are quartiles of cells per donor (`ind_cov`) in the full data, fixed before any CV split.
CELLECTION is trained on `disease_cov` with the folds in lupus_5folds.csv for seeds 0-9, and the
held-out predictions, pooled over folds and seeds, give one AUROC and one F1 per bin.

The score is the probability of "sle" (label column 1, after "healthy") thresholded at 0.5; the
per-fold MacroF1 is a true macro F1 over both classes, but the per-bin MacroF1 is the binary F1 of
"sle".

Reads $DATA_DIR/lupus/lupus_processed.h5ad (obs ind_cov, disease_cov) and lupus_5folds.csv. Writes to
$RESULTS_DIR/R1-q8/Results/lupus_disease/CELECTION/4-bin/: global_bin_edges.npy, donor_bin_map.csv
(ind_cov -> interval string), results_summary.csv (seed, fold, AUROC, MacroF1), per_bin_metrics.csv
(cell_bin, AUROC, MacroF1, n_donors = pooled donor x seed rows) and AUROC_per_bin.pdf /
F1_per_bin.pdf. Checkpoints go to ./checkpoints_lupus_disease_before/. Run this before
lupus_disease/downsample_and_retrain.py.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

import cellection

DECISION_THRESHOLD = 0.5

print(torch.cuda.is_available())

DATA_PATH = f"{DATA_DIR}/lupus"
adata = sc.read_h5ad(os.path.join(DATA_PATH, "lupus_processed.h5ad"))
folds = pd.read_csv(os.path.join(DATA_PATH, "lupus_5folds.csv"), index_col=0)

sample_key = "ind_cov"
task_key = "disease_cov"

num_folds = folds["fold"].nunique()

save_dir = (
    f"{RESULTS_DIR}/"
    "R1-q8/Results/lupus_disease/CELECTION/4-bin"
)
os.makedirs(save_dir, exist_ok=True)


# Quartiles of cells per donor. The interval strings saved in donor_bin_map.csv are the bin keys for
# downsample_and_retrain.py and plot_before_after.py.
all_counts = adata.obs.groupby(sample_key).size()

df = pd.DataFrame({"sample": all_counts.index, "count": all_counts.values})

df["bins"], bin_edges = pd.qcut(
    df["count"], q=4, retbins=True, duplicates="drop"
)

print("Global bin edges:", bin_edges)

np.save(f"{save_dir}/global_bin_edges.npy", bin_edges)

sample_bin_map = pd.cut(
    all_counts, bins=bin_edges, include_lowest=True, right=True
).astype(str)

sample_bin_map.to_csv(f"{save_dir}/donor_bin_map.csv")

bin_intervals = list(pd.Categorical(sample_bin_map).categories)
print("Bin intervals:", bin_intervals)


results = []
all_donors = []


for SEED in range(10):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n=== Running SEED {SEED} ===")

    for fold in range(1, num_folds + 1):
        print(f"\n===== Fold {fold} (Seed {SEED}) =====")

        train_donors = folds.loc[folds["fold"] != fold, "samples"]
        test_donors = folds.loc[folds["fold"] == fold, "samples"]

        adata_train = adata[adata.obs[sample_key].isin(train_donors)].copy()
        adata_test = adata[adata.obs[sample_key].isin(test_donors)].copy()

        if adata_train.obs[sample_key].nunique() <= 2:
            print("Skipping fold: too few donors")
            continue

        obj = cellection.cellectiion_object(
            adata=adata_train,
            task_type="classification",
            task_key=task_key,
            sample_key=sample_key,
            batch_key=None,
            model_type="classification",
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
            seed=SEED,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            hidden_layer=[256, 64],
            activation=nn.ReLU(),
            layernorm=True,
            batchnorm=False,
            patience=20,
            dropout_rate=0.1,
            prioritized_cell_pct=0.1,
            attention_cutoff=10,
            save_model=True,
            save_path="./checkpoints_lupus_disease_before",
        )

        # One bag per donor, 20% held out for early stopping. b[2] holds the bag lengths, so
        # single-bag batches are dropped (and the batches stay fixed across epochs).
        obj.prepare()
        obj.train_loader = [b for b in obj.train_loader if len(b[2]) > 1]
        obj.train()

        obj.adata = adata_test
        obj.prepare_inference()
        _, true_label_df, pred_label_df, _, _, _ = obj.inference()

        # Label columns are alphabetical: 0 = healthy, 1 = sle.
        y_true = np.argmax(true_label_df.values, axis=1)
        y_prob = pred_label_df.values[:, 1]
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        auroc = roc_auc_score(y_true, y_prob)
        macrof1 = f1_score(y_true, y_pred, average="macro")

        print(f"Fold {fold}, Seed {SEED}: AUROC={auroc:.3f}, MacroF1={macrof1:.3f}")
        results.append({"seed": SEED, "fold": fold, "AUROC": auroc, "MacroF1": macrof1})

        bag_sizes = (
            adata_test.obs.groupby(sample_key).size().rename("num_cells").to_frame()
        )
        bag_sizes = bag_sizes.loc[pred_label_df.index]

        df_donor = pd.DataFrame({
            "seed": SEED,
            "fold": fold,
            "donor_id": pred_label_df.index,
            "true_label": y_true,
            "pred_prob": y_prob,
            "num_cells": bag_sizes["num_cells"].values,
            "cell_bin": [sample_bin_map[d] for d in pred_label_df.index],
        })

        all_donors.append(df_donor)


df_res = pd.DataFrame(results)
df_res.to_csv(f"{save_dir}/results_summary.csv", index=False)

print("\n==== FINAL RESULTS (Baseline) ====")
print(df_res.groupby("fold")[["AUROC", "MacroF1"]].mean())
print("Mean AUROC:", df_res["AUROC"].mean())
print("Mean MacroF1:", df_res["MacroF1"].mean())


df_all = pd.concat(all_donors, ignore_index=True)

# Predictions are pooled over seeds and folds. f1_score falls back to average="binary" here, so the
# per-bin MacroF1 is the F1 of "sle", unlike the per-fold value above.
bin_stats = []
for cell_bin, g in df_all.groupby("cell_bin"):
    if len(np.unique(g["true_label"])) < 2:
        continue
    auroc_bin = roc_auc_score(g["true_label"], g["pred_prob"])
    f1_bin = f1_score(g["true_label"], (g["pred_prob"] >= DECISION_THRESHOLD).astype(int))

    bin_stats.append({
        "cell_bin": cell_bin,
        "AUROC": auroc_bin,
        "MacroF1": f1_bin,
        "n_donors": len(g),
    })

bin_df = pd.DataFrame(bin_stats)

bin_df.to_csv(f"{save_dir}/per_bin_metrics.csv", index=False)
print("\n==== PER-BIN METRICS ====")
print(bin_df)


bin_df = bin_df.dropna(subset=["AUROC", "MacroF1"])

bin_df["cell_bin"] = pd.Categorical(
    bin_df["cell_bin"],
    categories=[str(b) for b in bin_intervals],
    ordered=True,
)
bin_df = bin_df.sort_values("cell_bin")

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
sns.barplot(data=bin_df, x="cell_bin", y="AUROC", color="skyblue")
plt.title("Baseline AUROC vs Cell Bin (Lupus Disease)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{save_dir}/AUROC_per_bin.pdf")
plt.show()

plt.figure(figsize=(7, 5))
sns.barplot(data=bin_df, x="cell_bin", y="MacroF1", color="salmon")
plt.title("Baseline MacroF1 vs Cell Bin (Lupus Disease)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{save_dir}/F1_per_bin.pdf")
plt.show()
