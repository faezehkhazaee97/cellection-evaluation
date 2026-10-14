"""Lupus SLE vs healthy: cap the donors of the largest bin, retrain, and score each original bin again.

Bins come from lupus_disease/before_downsampling.py. Bin-4 donors with more than target_n cells (the
mean of the bin-2 and bin-3 medians) are cut to target_n random cells in every seed and fold, among
the training donors before training and the test donors before inference. Labels (`disease_cov`), CV
(lupus_5folds.csv) and seeds 0-9 match the baseline.

The score is again the probability of "sle" at a 0.5 threshold, but here both the per-fold and the
per-bin MacroF1 are macro F1 over the two classes (the baseline's per-bin MacroF1 is the binary F1 of
"sle").

Reads the lupus h5ad and fold file from $DATA_DIR/lupus/ and the baseline's global_bin_edges.npy and
donor_bin_map.csv. Writes summary_downsampled.csv, per_bin_metrics.csv (read by
lupus_disease/plot_before_after.py), AUROC_per_bin.pdf and MacroF1_per_bin.pdf to
$RESULTS_DIR/R1-q8/Results/lupus_disease/CELECTION/4-bin_downsample/. Checkpoints go to
./checkpoints_lupus_disease_downsampled/.
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

base_dir = (
    f"{RESULTS_DIR}/"
    "R1-q8/Results/lupus_disease/CELECTION/4-bin"
)

global_bin_edges = np.load(f"{base_dir}/global_bin_edges.npy")
donor_bin_map = (
    pd.read_csv(f"{base_dir}/donor_bin_map.csv", index_col=0).iloc[:, 0].to_dict()
)
print("Loaded global bin edges:", global_bin_edges)

all_counts = adata.obs.groupby(sample_key).size()
original_bins = pd.cut(
    all_counts,
    bins=global_bin_edges,
    include_lowest=True,
    right=True,
)

bin_intervals = list(original_bins.cat.categories)

# The same edges on the same counts reproduce the interval strings in donor_bin_map.csv. The unpacking
# fails if qcut merged two quartiles.
bin1, bin2, bin3, bin4 = bin_intervals

med2 = all_counts[original_bins == bin2].median()
med3 = all_counts[original_bins == bin3].median()
target_n_global = int(round((med2 + med3) / 2))

print(f"Median bin2 = {med2}, median bin3 = {med3}")
print("Target_n for downsampling bin4 =", target_n_global)


def downsample_bin4(adata_sub, donors, donor_bin_map, target_n, rng, sample_key="ind_cov"):
    """Cap each bin-4 donor above `target_n` cells at `target_n` random cells (returns a copy).

    The bin is the donor's original one from `donor_bin_map`.
    """
    keep = []
    donor_ids = adata_sub.obs[sample_key].values

    for donor in donors:
        donor_idx = np.where(donor_ids == donor)[0]
        donor_bin = donor_bin_map.get(str(donor), donor_bin_map.get(donor))

        if donor_bin == str(bin4) and len(donor_idx) > target_n:
            chosen = rng.choice(donor_idx, size=target_n, replace=False)
            keep.extend(chosen.tolist())
        else:
            keep.extend(donor_idx.tolist())

    return adata_sub[keep].copy()


save_dir = (
    f"{RESULTS_DIR}/"
    "R1-q8/Results/lupus_disease/CELECTION/4-bin_downsample"
)
os.makedirs(save_dir, exist_ok=True)

results = []
all_donors = []

for SEED in range(10):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # One generator per seed, shared by train and test downsampling in all its folds.
    rng = np.random.default_rng(SEED)

    print(f"\n=== Running SEED {SEED} ===")
    for fold in range(1, num_folds + 1):
        print(f"\n===== Fold {fold} (Seed {SEED}) =====")

        train_donors = folds.loc[folds["fold"] != fold, "samples"]
        test_donors = folds.loc[folds["fold"] == fold, "samples"]

        adata_train = adata[adata.obs[sample_key].isin(train_donors)].copy()
        adata_test = adata[adata.obs[sample_key].isin(test_donors)].copy()

        if adata_train.obs[sample_key].nunique() <= 2:
            continue

        adata_train_sub = downsample_bin4(
            adata_train, train_donors, donor_bin_map, target_n_global, rng, sample_key
        )
        print(
            f"[Fold {fold} Seed {SEED}] Train: {adata_train.n_obs} -> {adata_train_sub.n_obs}"
        )

        obj = cellection.cellectiion_object(
            adata=adata_train_sub,
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
            dropout_rate=0.1,
            prioritized_cell_pct=0.1,
            attention_cutoff=10,
            patience=20,
            save_model=True,
            save_path="./checkpoints_lupus_disease_downsampled",
        )

        obj.prepare()
        # Same single-bag batch filter as before_downsampling.py (b[2] = bag lengths).
        obj.train_loader = [b for b in obj.train_loader if len(b[2]) > 1]
        obj.train()

        adata_test_sub = downsample_bin4(
            adata_test, test_donors, donor_bin_map, target_n_global, rng, sample_key
        )
        print(
            f"[Fold {fold} Seed {SEED}] Test: {adata_test.n_obs} -> {adata_test_sub.n_obs}"
        )

        obj.adata = adata_test_sub
        obj.prepare_inference()
        _, true_df, pred_df, _, _, _ = obj.inference()

        y_true = np.argmax(true_df.values, axis=1)
        y_prob = pred_df.values[:, 1]   # probability of "sle"
        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        auroc = roc_auc_score(y_true, y_prob)
        macrof1 = f1_score(y_true, y_pred, average="macro")

        print(f"Fold {fold}, Seed {SEED}: AUROC={auroc:.3f}, MacroF1={macrof1:.3f}")
        results.append({"seed": SEED, "fold": fold, "AUROC": auroc, "MacroF1": macrof1})

        bag_sizes = (
            adata_test_sub.obs.groupby(sample_key).size().rename("num_cells").to_frame()
        )
        bag_sizes = bag_sizes.loc[pred_df.index]

        df_donor = pd.DataFrame(
            {
                "seed": SEED,
                "fold": fold,
                "donor_id": pred_df.index,
                "true_label": y_true,
                "pred_prob": y_prob,
                "num_cells": bag_sizes["num_cells"].values,
                "cell_bin": [donor_bin_map.get(str(d), donor_bin_map.get(d)) for d in pred_df.index],
            }
        )
        all_donors.append(df_donor)

df_res = pd.DataFrame(results)
df_res.to_csv(f"{save_dir}/summary_downsampled.csv", index=False)

print("\n==== FINAL DOWNSAMPLED RESULTS ====")
print(df_res.groupby("fold")[["AUROC", "MacroF1"]].mean())
print("Mean AUROC:", df_res["AUROC"].mean())
print("Mean F1:", df_res["MacroF1"].mean())

df_all = pd.concat(all_donors, ignore_index=True)

bin_stats = []
for cell_bin, g in df_all.groupby("cell_bin"):
    if len(np.unique(g["true_label"])) < 2:
        continue

    y_true = g["true_label"].values
    y_prob = g["pred_prob"].values
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    auroc_bin = roc_auc_score(y_true, y_prob)
    f1_bin = f1_score(y_true, y_pred, average="macro")

    bin_stats.append(
        {
            "cell_bin": cell_bin,
            "AUROC": auroc_bin,
            "MacroF1": f1_bin,
            "n_donors": len(g),
        }
    )

bin_df = pd.DataFrame(bin_stats)
bin_df["cell_bin"] = pd.Categorical(
    bin_df["cell_bin"], categories=[str(b) for b in bin_intervals], ordered=True
)
bin_df = bin_df.sort_values("cell_bin")

bin_df.to_csv(f"{save_dir}/per_bin_metrics.csv", index=False)

print("\n==== PER-BIN DOWNSAMPLED METRICS ====")
print(bin_df)

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
sns.barplot(data=bin_df, x="cell_bin", y="AUROC", color="skyblue")
plt.title("AUROC vs Cell Bin (Downsampled) - Lupus Disease")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{save_dir}/AUROC_per_bin.pdf")
plt.show()

plt.figure(figsize=(7, 5))
sns.barplot(data=bin_df, x="cell_bin", y="MacroF1", color="salmon")
plt.title("MacroF1 vs Cell Bin (Downsampled) - Lupus Disease")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{save_dir}/MacroF1_per_bin.pdf")
plt.show()
