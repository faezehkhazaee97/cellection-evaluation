"""Ziegler COVID-19, second half: cap the two largest cell-count bins, retrain, score each bin again.

The 8 edges saved by covid/before_downsampling.py are re-applied to the full-data cell counts, which
gives back the same donor -> bin map. target_n is the mean of the median cell counts of bins 5 and 6.
In every seed and fold, each bin-7 or bin-8 donor with more than target_n cells is cut to target_n
random cells, among the training donors (so the model is retrained on the smaller bags) and among the
test donors before inference. Results stay grouped by each donor's original bin, so the before and
after bars cover the same donors.

Labels, model settings, CV and seeds 0-9 are those of the baseline, and so are the metrics: label
column 1 (alphabetical, "control" for a derived `disease_id`) is the positive class and the MacroF1
column is its binary F1 at a 0.5 threshold.

Reads covid_Ziegler_processed.h5ad and covid_5folds.csv from $DATA_DIR/covid/ and
bin_8/global_bin_edges.npy from the baseline. Writes to
$RESULTS_DIR/R1-q8/Results/covid_Ziegler/CELECTION/bin_8_subsample_retrain/:
  donor_bin_map_original.csv                      donor_id -> original interval string
  cellection_results_summary.csv                  seed, fold, AUROC, MacroF1
  cellection_global_bins_retrain_downsampled.csv  cell_bin, AUROC, MacroF1, n_donors per original
                                                  bin; read by covid/plot_before_after.py
  global_auroc_vs_cell_bin_retrain.pdf, global_macrof1_vs_cell_bin_retrain.pdf
Checkpoints go to ./checkpoints_covid_downsampled/.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

import cellection

DECISION_THRESHOLD = 0.5

print(torch.cuda.is_available())

adata = sc.read_h5ad(
    f"{DATA_DIR}/covid/covid_Ziegler_processed.h5ad"
)
folds = pd.read_csv(
    f"{DATA_DIR}/covid/covid_5folds.csv", index_col=0
)  # columns: samples (donor_id), fold

if "disease_id" not in adata.obs.columns:
    adata.obs["disease_id"] = "control"
    adata.obs.loc[
        adata.obs["disease__ontology_label"].isin(["COVID-19", "long COVID-19"]),
        "disease_id"
    ] = "case"
    print("Added disease_id column ")

sample_key = "donor_id"
task_key = "disease_id"
num_folds = folds["fold"].nunique()

global_bin_edges = np.load(
    f"{RESULTS_DIR}/R1-q8/Results/covid_Ziegler/CELECTION/bin_8/global_bin_edges.npy"
)
print("Loaded global fixed bin edges:", global_bin_edges)

all_counts = adata.obs.groupby(sample_key).size().rename("num_cells")

original_bins = pd.cut(
    all_counts,
    bins=global_bin_edges,
    include_lowest=True,
    right=True,
    duplicates="drop",
)
donor_bin_map = original_bins.astype(str).to_dict()

print("Bin counts (original, full dataset):")
print(original_bins.value_counts().sort_index())

save_dir = f"{RESULTS_DIR}/R1-q8/Results/covid_Ziegler/CELECTION/bin_8_subsample_retrain"
os.makedirs(save_dir, exist_ok=True)

pd.Series(donor_bin_map).to_csv(f"{save_dir}/donor_bin_map_original.csv")

def downsample_bin7_8_for_fold(
    adata_sub,
    donors_in_fold,
    all_counts,
    original_bins,
    bin7_interval,
    bin8_interval,
    target_n,
    rng,
    sample_key="donor_id",
):
    """Cap each bin-7/8 donor above `target_n` cells at `target_n` random cells (returns a copy).

    Bin and cell count come from the full data (`original_bins`, `all_counts`), not from `adata_sub`.
    """
    keep_indices = []
    donor_ids = adata_sub.obs[sample_key].values

    for donor in donors_in_fold:
        n_orig = all_counts.loc[donor]
        donor_bin = original_bins.loc[donor]
        donor_idx = np.where(donor_ids == donor)[0]

        if donor_bin in [bin7_interval, bin8_interval] and n_orig > target_n:
            chosen = rng.choice(donor_idx, size=target_n, replace=False)
            keep_indices.extend(chosen.tolist())
        else:
            keep_indices.extend(donor_idx.tolist())

    return adata_sub[keep_indices].copy()


results = []
all_donors = []

cats = list(original_bins.cat.categories)

# Categories run from the smallest donors up, so cats[k] is bin k+1. Capping at the mean of the bin-5
# and bin-6 medians shrinks bins 7-8 to roughly the size of the bins just below them.
bin5_interval = cats[4]
bin6_interval = cats[5]
bin7_interval = cats[6]
bin8_interval = cats[7]
med5_global = all_counts[original_bins == bin5_interval].median()
med6_global = all_counts[original_bins == bin6_interval].median()
target_n_global = int(round((med5_global + med6_global) / 2.0))

print(f"Global med5={med5_global}, med6={med6_global}, target_n={target_n_global}")


for SEED in range(10):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # One generator per seed, shared by train and test downsampling in all its folds, so which cells
    # survive depends on the seed and on the fold order.
    rng = np.random.default_rng(SEED)

    print(f"\n Running SEED {SEED}")
    for fold in range(1, num_folds + 1):
        print(f"\n===== Fold {fold} (Seed {SEED}) =====")

        train_donors = folds.loc[folds["fold"] != fold, "samples"]
        test_donors = folds.loc[folds["fold"] == fold, "samples"]

        adata_train = adata[adata.obs[sample_key].isin(train_donors)].copy()
        adata_test = adata[adata.obs[sample_key].isin(test_donors)].copy()

        n_donors = adata_train.obs[sample_key].nunique()
        if n_donors <= 2:
            print(f"Skipping fold {fold}: only {n_donors} donors.")
            continue

        adata_train_sub = downsample_bin7_8_for_fold(
            adata_sub=adata_train,
            donors_in_fold=train_donors,
            all_counts=all_counts,
            original_bins=original_bins,
            bin7_interval=bin7_interval,
            bin8_interval=bin8_interval,
            target_n=target_n_global,
            rng=rng,
            sample_key=sample_key,
        )

        print(
            f"[Fold {fold} Seed {SEED}] Train before: {adata_train.n_obs} cells, "
            f"after downsampling: {adata_train_sub.n_obs} cells"
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
            save_model=True,
            save_path="./checkpoints_covid_downsampled",
        )
        obj.prepare()
        # Same single-bag batch filter as before_downsampling.py (b[2] = bag lengths).
        obj.train_loader = [b for b in obj.train_loader if len(b[2]) > 1]
        obj.train()

        adata_test_sub = downsample_bin7_8_for_fold(
            adata_sub=adata_test,
            donors_in_fold=test_donors,
            all_counts=all_counts,
            original_bins=original_bins,
            bin7_interval=bin7_interval,
            bin8_interval=bin8_interval,
            target_n=target_n_global,
            rng=rng,
            sample_key=sample_key,
        )

        obj.adata = adata_test_sub
        obj.prepare_inference()
        sample_metadata, true_label_df, pred_label_df, _, _, _ = obj.inference()

        y_true = np.argmax(true_label_df.values, axis=1)

        y_prob = pred_label_df.values[:, 1]

        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        # Binary F1 of column 1, stored under the name MacroF1
        auroc = roc_auc_score(y_true, y_prob)
        macrof1 = f1_score(y_true, y_pred, average="binary")

        print(f"Fold {fold}, Seed {SEED}: AUROC={auroc:.3f}, MacroF1={macrof1:.3f}")
        results.append({"seed": SEED, "fold": fold, "AUROC": auroc, "MacroF1": macrof1})

        bag_sizes = (
            adata_test_sub.obs.groupby(sample_key)
            .size()
            .rename("num_cells_downsampled")
            .to_frame()
        )
        bag_sizes = bag_sizes.loc[pred_label_df.index]

        orig_sizes = all_counts.loc[pred_label_df.index].rename("num_cells_original")

        y_true_cls = np.argmax(true_label_df.values, axis=1)
        y_pred_prob = pred_label_df.values[:, 1]

        df_donor = pd.DataFrame({
            "seed": SEED,
            "fold": fold,
            "donor_id": pred_label_df.index,
            "true_label": y_true_cls,
            "pred_prob": y_pred_prob,
            "num_cells_downsampled": bag_sizes["num_cells_downsampled"].values,
            "num_cells_original": orig_sizes.values,
        })

        # A capped bin-7/8 donor is still reported under its original bin.
        df_donor["cell_bin"] = df_donor["donor_id"].map(donor_bin_map)
        all_donors.append(df_donor)

df_res = pd.DataFrame(results)
df_res.to_csv(f"{save_dir}/cellection_results_summary.csv", index=False)

print("\n==== Final Results ====")
print(df_res.groupby("fold")[["AUROC", "MacroF1"]].mean())
print("\nMean AUROC:", df_res["AUROC"].mean())
print("Mean MacroF1:", df_res["MacroF1"].mean())

df_all = pd.concat(all_donors, ignore_index=True)
print(f"\nTotal test donors across folds/seeds: {len(df_all)}")

bin_stats = []
for cell_bin, group in df_all.groupby("cell_bin"):
    if len(np.unique(group["true_label"])) < 2:
        print(f"Skipping bin {cell_bin} — single class only.")
        continue

    y_true_bin = group["true_label"].values.astype(int)
    y_prob_bin = group["pred_prob"].values
    y_pred_bin = (y_prob_bin >= DECISION_THRESHOLD).astype(int)

    auroc_bin = roc_auc_score(y_true_bin, y_prob_bin)
    f1_bin = f1_score(y_true_bin, y_pred_bin, average="binary")

    bin_stats.append({
        "cell_bin": str(cell_bin),
        "AUROC": auroc_bin,
        "MacroF1": f1_bin,
        "n_donors": len(group)
    })

bin_df = pd.DataFrame(bin_stats)
bin_df.to_csv(f"{save_dir}/cellection_global_bins_retrain_downsampled.csv", index=False)

print("\n==== Global per-bin metrics (original bins, retrained on downsampled data) ====")
print(bin_df)

interval_order = [str(iv) for iv in original_bins.cat.categories]

bin_df["cell_bin"] = pd.Categorical(
    bin_df["cell_bin"],
    categories=interval_order,
    ordered=True
)
import seaborn as sns
import matplotlib.pyplot as plt

bin_df = bin_df.dropna(subset=["AUROC", "MacroF1"])

plt.figure(figsize=(7,5))
sns.barplot(data=bin_df, x="cell_bin", y="AUROC")
plt.title("Global AUROC vs. Original Cell Bin (Retrained on Downsampled Data)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{save_dir}/global_auroc_vs_cell_bin_retrain.pdf")
plt.show()

plt.figure(figsize=(7,5))
sns.barplot(data=bin_df, x="cell_bin", y="MacroF1")
plt.title("Global Macro-F1 vs. Original Cell Bin (Retrained on Downsampled Data)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{save_dir}/global_macrof1_vs_cell_bin_retrain.pdf")
plt.show()
