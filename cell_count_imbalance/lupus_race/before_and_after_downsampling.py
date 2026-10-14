
"""Lupus, Asian vs White: the cell-count bin test, baseline and downsampled retraining in one file.

Stage 1 bins donors (`ind_cov`) into quartiles of cells per donor on the full data, trains CELLECTION
on all cells for `pop_cov` with the folds in race_5folds.csv and seeds 0-9, and scores the pooled
held-out predictions per bin. Stage 2 reloads those bins, cuts bin-4 donors above target_n cells (mean
of the bin-2 and bin-3 medians) to target_n random cells in training and in test, retrains, and
reports the same metrics under the original bins.

The score is the probability of "WHITE" (label column 1, after "ASIAN") at a 0.5 threshold; in both
stages the per-fold MacroF1 is a macro F1, but the per-bin MacroF1 is the binary F1 of "WHITE".

Reads $DATA_DIR/lupus/lupus_processed.h5ad (obs ind_cov, pop_cov) and race_5folds.csv. Writes to
$RESULTS_DIR/R1-q8/Results/lupus_race/CELECTION/:
  4-bin/             global_bin_edges.npy, donor_bin_map.csv (ind_cov -> interval string),
                     results_summary.csv (seed, fold, AUROC, MacroF1), per_bin_metrics.csv (cell_bin,
                     AUROC, MacroF1, n_donors), AUROC_per_bin.pdf, F1_per_bin.pdf
  4-bin_downsample/  summary_downsampled.csv, per_bin_metrics.csv, AUROC_per_bin.pdf,
                     MacroF1_per_bin.pdf
Checkpoints go to ./checkpoints_lupus_race_before/ and ./checkpoints_lupus_race_downsampled/.
lupus_race/plot_before_after.py draws the comparison.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import matplotlib.pyplot as plt
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

# Stage 1: baseline on all cells.
adata = sc.read_h5ad(
    f"{DATA_DIR}/lupus/lupus_processed.h5ad"
)
folds = pd.read_csv(
    f"{DATA_DIR}/lupus/race_5folds.csv",
    index_col=0
)

sample_key = "ind_cov"   # donor ID
task_key = "pop_cov"     # race (ASIAN vs WHITE)

num_folds = folds["fold"].nunique()

save_dir = (
    f"{RESULTS_DIR}/"
    "R1-q8/Results/lupus_race/CELECTION/4-bin"
)
os.makedirs(save_dir, exist_ok=True)

# Quartiles of cells per donor on the full data. The interval strings are the bin keys in stage 2 and
# in plot_before_after.py, so a donor's bin never depends on the fold or seed.
all_counts = adata.obs.groupby(sample_key).size()

df = pd.DataFrame({"donor": all_counts.index, "count": all_counts.values})

df["bins"], bin_edges = pd.qcut(
    df["count"], q=4, retbins=True, duplicates="drop"
)

print("Global bin edges:", bin_edges)

np.save(f"{save_dir}/global_bin_edges.npy", bin_edges)

donor_bin_map = pd.cut(
    all_counts, bins=bin_edges, include_lowest=True, right=True
).astype(str)

donor_bin_map.to_csv(f"{save_dir}/donor_bin_map.csv")

bin_intervals = list(pd.Categorical(donor_bin_map).categories)
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
            save_path="./checkpoints_lupus_race_before",
        )

        # One bag per donor, 20% held out for early stopping. b[2] holds the bag lengths, so
        # single-bag batches are dropped (and the batches stay fixed across epochs).
        obj.prepare()
        obj.train_loader = [b for b in obj.train_loader if len(b[2]) > 1]
        obj.train()

        obj.adata = adata_test
        obj.prepare_inference()
        sample_meta, true_label_df, pred_label_df, _, _, _ = obj.inference()

        # Label columns are alphabetical: 0 = ASIAN, 1 = WHITE.
        y_true = np.argmax(true_label_df.values, axis=1)
        y_score = pred_label_df.values[:, 1]
        y_pred = (y_score >= DECISION_THRESHOLD).astype(int)

        auroc = roc_auc_score(y_true, y_score)
        macrof1 = f1_score(y_true, y_pred, average="macro")

        print(f"Fold {fold}, Seed {SEED}: AUROC={auroc:.3f}, F1={macrof1:.3f}")
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
            "pred_prob": y_score,
            "num_cells": bag_sizes["num_cells"].values,
            "cell_bin": [donor_bin_map[d] for d in pred_label_df.index],
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
# per-bin MacroF1 is the F1 of "WHITE".
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
    ordered=True
)
bin_df = bin_df.sort_values("cell_bin")

plt.figure(figsize=(7,5))
sns.barplot(data=bin_df, x="cell_bin", y="AUROC", color="skyblue")
plt.title("Baseline AUROC vs Cell Bin (Lupus Race)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{save_dir}/AUROC_per_bin.pdf")
plt.show()

plt.figure(figsize=(7,5))
sns.barplot(data=bin_df, x="cell_bin", y="MacroF1", color="salmon")
plt.title("Baseline MacroF1 vs Cell Bin (Lupus Race)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{save_dir}/F1_per_bin.pdf")
plt.show()


# Stage 2: cap bin-4 donors and retrain. It re-imports, reloads the data and reads the stage-1 bins
# back from disk, as if it were a separate script.
import cellection
import scanpy as sc
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

print(torch.cuda.is_available())

adata = sc.read_h5ad(
    f"{DATA_DIR}/lupus/lupus_processed.h5ad"
)
folds = pd.read_csv(
    f"{DATA_DIR}/lupus/race_5folds.csv",
    index_col=0
)

sample_key = "ind_cov"   # donors
task_key = "pop_cov"     # race label
num_folds = folds["fold"].nunique()

base_dir = (
    f"{RESULTS_DIR}/"
    "R1-q8/Results/lupus_race/CELECTION/4-bin"
)

global_bin_edges = np.load(f"{base_dir}/global_bin_edges.npy")
donor_bin_map = (
    pd.read_csv(f"{base_dir}/donor_bin_map.csv", index_col=0)
      .iloc[:, 0]
      .to_dict()
)
print("Loaded global bin edges:", global_bin_edges)

# Stage-1 edges on the full-data counts reproduce the interval strings in donor_bin_map.csv.
all_counts = adata.obs.groupby(sample_key).size()
original_bins = pd.cut(
    all_counts,
    bins=global_bin_edges,
    include_lowest=True,
    right=True,
)
bin_intervals = list(original_bins.cat.categories)

# The unpacking fails if qcut merged two quartiles.
bin1, bin2, bin3, bin4 = bin_intervals

med2 = all_counts[original_bins == bin2].median()
med3 = all_counts[original_bins == bin3].median()
target_n_global = int(round((med2 + med3) / 2))

print(f"Median bin2 = {med2}, median bin3 = {med3}")
print("Target_n for downsampling bin4 =", target_n_global)

def downsample_bin4(
    adata_sub, donors, donor_bin_map, target_n, rng, sample_key="ind_cov"
):
    """Cap each bin-4 donor above `target_n` cells at `target_n` random cells (returns a copy).

    The bin is the donor's original stage-1 bin from `donor_bin_map`.
    """

    keep = []
    donor_ids = adata_sub.obs[sample_key].values

    for donor in donors:

        donor_idx = np.where(donor_ids == donor)[0]
        donor_bin = donor_bin_map[str(donor)]

        if donor_bin == str(bin4) and len(donor_idx) > target_n:
            chosen = rng.choice(donor_idx, size=target_n, replace=False)
            keep.extend(chosen.tolist())
        else:
            keep.extend(donor_idx.tolist())

    return adata_sub[keep].copy()


save_dir = (
    f"{RESULTS_DIR}/"
    "R1-q8/Results/lupus_race/CELECTION/4-bin_downsample"
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
        test_donors  = folds.loc[folds["fold"] == fold, "samples"]

        adata_train = adata[adata.obs[sample_key].isin(train_donors)].copy()
        adata_test  = adata[adata.obs[sample_key].isin(test_donors)].copy()

        if adata_train.obs[sample_key].nunique() <= 2:
            continue

        adata_train_sub = downsample_bin4(
            adata_train, train_donors, donor_bin_map,
            target_n_global, rng, sample_key
        )
        print(
            f"[Fold {fold} Seed {SEED}] Train: {adata_train.n_obs} → {adata_train_sub.n_obs}"
        )

        obj = cellection.cellectiion_object(
            adata=adata_train_sub,
            task_type="classification",
            task_key=task_key,
            sample_key=sample_key,
            batch_key=None,
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
            dropout_rate=0.1,
            patience=20,
            save_model=True,
            save_path="./checkpoints_lupus_race_downsampled",
        )

        obj.prepare()
        # Same single-bag batch filter as stage 1.
        obj.train_loader = [b for b in obj.train_loader if len(b[2]) > 1]
        obj.train()

        adata_test_sub = downsample_bin4(
            adata_test, test_donors, donor_bin_map,
            target_n_global, rng, sample_key
        )
        print(
            f"[Fold {fold} Seed {SEED}] Test: {adata_test.n_obs} → {adata_test_sub.n_obs}"
        )

        obj.adata = adata_test_sub
        obj.prepare_inference()

        sample_meta, true_label_df, pred_label_df, _, _, _ = obj.inference()

        y_true = np.argmax(true_label_df.values, axis=1)
        y_score = pred_label_df.values[:, 1]
        y_pred  = (y_score >= DECISION_THRESHOLD).astype(int)

        auroc = roc_auc_score(y_true, y_score)
        macrof1 = f1_score(y_true, y_pred, average="macro")

        print(f"Fold {fold}, Seed {SEED}: AUROC={auroc:.3f}, MacroF1={macrof1:.3f}")

        results.append(
            {"seed": SEED, "fold": fold, "AUROC": auroc, "MacroF1": macrof1}
        )

        bag_sizes = (
            adata_test_sub.obs.groupby(sample_key).size().rename("num_cells").to_frame()
        )
        bag_sizes = bag_sizes.loc[pred_label_df.index]

        df_donor = pd.DataFrame({
            "seed": SEED,
            "fold": fold,
            "donor_id": pred_label_df.index,
            "true_label": y_true,
            "pred_prob": y_score,
            "num_cells": bag_sizes["num_cells"].values,
            "cell_bin": [donor_bin_map[str(d)] for d in pred_label_df.index],
        })
        all_donors.append(df_donor)

df_res = pd.DataFrame(results)
df_res.to_csv(f"{save_dir}/summary_downsampled.csv", index=False)

print("\n==== FINAL DOWNSAMPLED RESULTS ====")
print(df_res.groupby("fold")[["AUROC","MacroF1"]].mean())
print("Mean AUROC:", df_res["AUROC"].mean())
print("Mean F1:", df_res["MacroF1"].mean())

df_all = pd.concat(all_donors, ignore_index=True)

bin_stats = []
for cell_bin, g in df_all.groupby("cell_bin"):

    if len(np.unique(g["true_label"])) < 2:
        continue

    auroc_bin = roc_auc_score(g["true_label"], g["pred_prob"])
    # average="binary" by default: the F1 of "WHITE", as in stage 1
    f1_bin = f1_score(g["true_label"], (g["pred_prob"] >= DECISION_THRESHOLD).astype(int))

    bin_stats.append({
        "cell_bin": cell_bin,
        "AUROC": auroc_bin,
        "MacroF1": f1_bin,
        "n_donors": len(g)
    })

bin_df = pd.DataFrame(bin_stats)

bin_df["cell_bin"] = pd.Categorical(
    bin_df["cell_bin"],
    categories=[str(b) for b in bin_intervals],
    ordered=True
)
bin_df = bin_df.sort_values("cell_bin")

bin_df.to_csv(f"{save_dir}/per_bin_metrics.csv", index=False)

print("\n==== PER-BIN DOWNSAMPLED METRICS ====")
print(bin_df)

plt.figure(figsize=(7,5))
sns.barplot(data=bin_df, x="cell_bin", y="AUROC", color="skyblue")
plt.title("AUROC vs Cell Bin (Downsampled) — Lupus")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{save_dir}/AUROC_per_bin.pdf")
plt.show()

plt.figure(figsize=(7,5))
sns.barplot(data=bin_df, x="cell_bin", y="MacroF1", color="salmon")
plt.title("MacroF1 vs Cell Bin (Downsampled) — Lupus")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{save_dir}/MacroF1_per_bin.pdf")
plt.show()
