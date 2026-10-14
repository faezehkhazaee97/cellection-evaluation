
"""Does the number of cells per donor change how well CELLECTION classifies Ziegler COVID-19 donors?

Donors are split into 8 quantile bins by their cell count in the full dataset (pd.qcut, edges
rounded to integers). The edges are fixed before any CV split, so a donor keeps its bin in every
fold and seed and in covid/downsample_and_retrain.py. CELLECTION (gated-attention MIL, case vs
control) is trained on all cells with the 5-fold donor CV in covid_5folds.csv for seeds 0-9; the
held-out donor predictions, pooled over folds and seeds, give one AUROC and one F1 per bin.

The label is obs `disease_id`. When that column is missing it is derived from
`disease__ontology_label` ("COVID-19" and "long COVID-19" -> "case", anything else -> "control").
Both metrics treat label column 1 (alphabetical, so "control" for the derived label) as positive, and
the MacroF1 column is really the binary F1 of that class at a 0.5 threshold.

Reads $DATA_DIR/covid/covid_Ziegler_processed.h5ad (obs donor_id and the label) and covid_5folds.csv.
Writes to $RESULTS_DIR/R1-q8/Results/covid_Ziegler/CELECTION/bin_8/:
  global_bin_edges.npy            the 9 integer edges, reloaded by covid/downsample_and_retrain.py
  donor_bin_map.csv               donor_id -> interval string such as "(86.0, 120.0]"
  cellection_results_summary.csv  seed, fold, AUROC, MacroF1
  cellection_global_bins.csv      cell_bin, AUROC, MacroF1, n_donors (pooled donor x seed rows);
                                  read by covid/plot_before_after.py
  global_auroc_vs_cell_bin.pdf, global_macrof1_vs_cell_bin.pdf
Checkpoints go to ./checkpoints_covid_before/. Run this before covid/downsample_and_retrain.py.
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
save_dir = f"{RESULTS_DIR}/R1-q8/Results/covid_Ziegler/CELECTION/bin_8"
os.makedirs(save_dir, exist_ok=True)

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

all_counts = adata.obs.groupby(sample_key).size()

# qcut gives bins with about equal numbers of donors, not of cells; duplicates="drop" would merge tied
# quantiles and leave fewer than 8 bins. include_lowest keeps the smallest donor in bin 1 (pandas
# prints that edge as e.g. 17.999). The interval strings are the bin keys in every later script.
_, global_bin_edges = pd.qcut(all_counts, q=8, retbins=True, duplicates="drop")
global_bin_edges = np.round(global_bin_edges).astype(int)
np.save(f"{save_dir}/global_bin_edges.npy", global_bin_edges)

original_bins = pd.cut(
    all_counts,
    bins=global_bin_edges,
    include_lowest=True,
    right=True
)
donor_bin_map = original_bins.astype(str).to_dict()

pd.Series(donor_bin_map).to_csv(f"{save_dir}/donor_bin_map.csv")


results = []
all_donors = []

for SEED in range(10):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n Running SEED {SEED}")
    for fold in range(1, num_folds + 1):
        print(f"\n===== Fold {fold} (Seed {SEED}) =====")

        train_donors = folds.loc[folds["fold"] != fold, "samples"]
        test_donors = folds.loc[folds["fold"] == fold, "samples"]

        adata_train = adata[adata.obs[sample_key].isin(train_donors)]
        adata_test = adata[adata.obs[sample_key].isin(test_donors)].copy()

        n_donors = adata_train.obs[sample_key].nunique()
        if n_donors <= 2:
            print(f" Skipping fold {fold}: only {n_donors} donors.")
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
            save_path="./checkpoints_covid_before",
        )

        # prepare() makes one bag per donor and holds out 20% of the training bags for early stopping;
        # train() reloads the best checkpoint at the end.
        obj.prepare()
        # Batches are (padded_bags, labels, bag_lengths, cell_ids), so this drops single-bag batches.
        # Turning the loader into a list also freezes the batches: every epoch sees the same ones.
        obj.train_loader = [b for b in obj.train_loader if len(b[2]) > 1]
        obj.train()

        obj.adata = adata_test
        obj.prepare_inference()
        # Every cell id in the inference bags should come from adata_test.
        cell_ids = [cell for bag in obj.mil_dataset.meta_ids for cell in bag]

        valid = pd.Index(cell_ids).intersection(adata_test.obs.index)

        if len(valid) != len(cell_ids):
            print("WARNING: Some cell IDs generated by MIL do not exist in adata_test!")
            print("Missing IDs:", set(cell_ids) - set(valid))

        sample_metadata, true_label_df, pred_label_df, _, _, _ = obj.inference()
        # One row per test donor (index donor_id), one column per class in alphabetical order;
        # pred_label_df holds softmax probabilities.

        y_true = np.argmax(true_label_df.values, axis=1)

        y_prob = pred_label_df.values[:, 1]

        y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

        # Binary F1 of column 1, stored under the name MacroF1
        auroc = roc_auc_score(y_true, y_prob)
        macrof1 = f1_score(y_true, y_pred, average="binary")

        print(f"Fold {fold}, Seed {SEED}: AUROC={auroc:.3f}, MacroF1={macrof1:.3f}")
        results.append({"seed": SEED, "fold": fold, "AUROC": auroc, "MacroF1": macrof1})

        bag_sizes = (
            adata_test.obs.groupby(sample_key)
            .size()
            .rename("num_cells")
            .to_frame()
        )
        bag_sizes = bag_sizes.loc[pred_label_df.index]
        y_true_cls = np.argmax(true_label_df.values, axis=1)
        y_pred_prob = pred_label_df.values[:, 1] if pred_label_df.shape[1] > 1 else pred_label_df.values[:, 0]

        df_donor = pd.DataFrame({
            "seed": SEED,
            "fold": fold,
            "donor_id": pred_label_df.index,
            "true_label": y_true_cls,
            "pred_prob": y_pred_prob,
            "num_cells": bag_sizes["num_cells"].values,
        })
        all_donors.append(df_donor)

df_res = pd.DataFrame(results)
df_res.to_csv(f"{save_dir}/cellection_results_summary.csv", index=False)

print("\n==== Final Results ====")
print(df_res.groupby("fold")[["AUROC", "MacroF1"]].mean())
print("\nMean AUROC:", df_res["AUROC"].mean())
print("Mean MacroF1:", df_res["MacroF1"].mean())

# Each donor is tested once per seed, so df_all has n_donors x 10 rows. They are pooled per bin into
# one AUROC and one F1 (no per-seed spread), and single-class bins are skipped.
df_all = pd.concat(all_donors, ignore_index=True)
print(f"\nTotal test donors across folds/seeds: {len(df_all)}")

df_all["cell_bin"] = df_all["donor_id"].map(donor_bin_map)


bin_stats = []
for cell_bin, group in df_all.groupby("cell_bin"):
    if len(np.unique(group["true_label"])) < 2:
        print(f"Skipping bin {cell_bin} — single class only.")
        continue
    auroc_bin = roc_auc_score(group["true_label"], group["pred_prob"])
    f1_bin = f1_score(group["true_label"], (group["pred_prob"] >= DECISION_THRESHOLD).astype(int))
    bin_stats.append({
        "cell_bin": str(cell_bin),
        "AUROC": auroc_bin,
        "MacroF1": f1_bin,
        "n_donors": len(group)
    })

bin_df = pd.DataFrame(bin_stats)
bin_df.to_csv(f"{save_dir}/cellection_global_bins.csv", index=False)

print("\n==== Global per-bin metrics ====")
print(bin_df)

import seaborn as sns
import matplotlib.pyplot as plt

bin_df = bin_df.dropna(subset=["AUROC", "MacroF1"])
interval_order = [str(iv) for iv in original_bins.cat.categories]

bin_df["cell_bin"] = pd.Categorical(
    bin_df["cell_bin"],
    categories=interval_order,
    ordered=True
)
plt.figure(figsize=(7,5))
sns.barplot(data=bin_df, x="cell_bin", y="AUROC", color="skyblue")
plt.title("Global AUROC vs. Number of Cells per Donor (binned across all folds/seeds)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{save_dir}/global_auroc_vs_cell_bin.pdf")
plt.show()

plt.figure(figsize=(7,5))
sns.barplot(data=bin_df, x="cell_bin", y="MacroF1", color="salmon")
plt.title("Global Macro-F1 vs. Number of Cells per Donor (binned across all folds/seeds)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{save_dir}/global_macrof1_vs_cell_bin.pdf")
plt.show()
