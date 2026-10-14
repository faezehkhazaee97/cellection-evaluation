"""Cardio, one bag per donor: the 4-bin version of the cell-count test in covid/before_downsampling.py.

Donors are split into 4 quantile bins of cells per donor. The edges are computed once on the full
data and saved, so a donor's bin never depends on the fold or seed. CELLECTION is trained on all cells
for the 3-class `disease` label (DCM / HCM / NF) with the folds in cardio_5folds.csv and seeds 0-9.
Held-out predictions, pooled over folds and seeds, give a macro one-vs-rest AUROC and a macro F1
(argmax prediction) per bin.

Reads $DATA_DIR/cardio/cardio_processed.h5ad (obs donor_id, disease) and cardio_5folds.csv. Writes to
$RESULTS_DIR/R1-q8/Results/cardio/CELECTION/4-bin/: global_bin_edges.npy (5 edges) and
donor_bin_map.csv for the downsampling script, results_summary.csv (seed, fold, AUROC, MacroF1),
per_bin_metrics.csv (cell_bin, AUROC, MacroF1, n_donors = pooled donor x seed rows) for the plot
script, and AUROC_per_bin.pdf / F1_per_bin.pdf. Checkpoints go to ./checkpoints_cardio_before/.
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

print(torch.cuda.is_available())

DATA_PATH = f"{DATA_DIR}/cardio"
adata = sc.read_h5ad(os.path.join(DATA_PATH, "cardio_processed.h5ad"))
folds = pd.read_csv(os.path.join(DATA_PATH, "cardio_5folds.csv"), index_col=0)

sample_key = "donor_id"
task_key = "disease"

num_folds = folds["fold"].nunique()

save_dir = (
    f"{RESULTS_DIR}/"
    "R1-q8/Results/cardio/CELECTION/4-bin"
)
os.makedirs(save_dir, exist_ok=True)


def macro_auroc(true_df, pred_df):
    """Unweighted mean of one-vs-rest AUROCs over the class columns of one-hot `true_df`.

    Classes whose column is constant here (absent, or the only class present) are skipped.
    """
    scores = []
    for c in true_df.columns:
        if len(np.unique(true_df[c])) < 2:
            continue
        scores.append(roc_auc_score(true_df[c], pred_df[c]))
    return float(np.mean(scores)) if scores else float("nan")


# Quartiles of cells per donor, so each bin holds about the same number of donors. The interval
# strings, e.g. "(7209.999, 10181.0]", are the bin keys in donor_bin_map.csv and in the later scripts.
all_counts = adata.obs.groupby(sample_key).size()
df = pd.DataFrame({"donor": all_counts.index, "count": all_counts.values})
df["bins"], bin_edges = pd.qcut(df["count"], q=4, retbins=True, duplicates="drop")

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
            save_path="./checkpoints_cardio_before",
        )

        # One bag per donor, 20% of the training bags held out for early stopping. b[2] holds the bag
        # lengths, so single-bag batches are dropped (and the batches stay fixed across epochs).
        obj.prepare()
        obj.train_loader = [b for b in obj.train_loader if len(b[2]) > 1]
        obj.train()

        obj.adata = adata_test
        obj.prepare_inference()
        _, true_df, pred_df, _, _, _ = obj.inference()

        y_true = np.argmax(true_df.values, axis=1)
        y_pred = np.argmax(pred_df.values, axis=1)

        auroc = macro_auroc(true_df, pred_df)
        macrof1 = f1_score(y_true, y_pred, average="macro")

        print(f"Fold {fold}, Seed {SEED}: AUROC={auroc:.3f}, MacroF1={macrof1:.3f}")
        results.append({"seed": SEED, "fold": fold, "AUROC": auroc, "MacroF1": macrof1})

        bag_sizes = (
            adata_test.obs.groupby(sample_key).size().rename("num_cells").to_frame()
        )
        bag_sizes = bag_sizes.loc[pred_df.index]

        df_donor = pd.DataFrame(
            {
                "seed": SEED,
                "fold": fold,
                "donor_id": pred_df.index,
                "true_label": y_true,
                "num_cells": bag_sizes["num_cells"].values,
                "cell_bin": [donor_bin_map[d] for d in pred_df.index],
            }
        )
        # prob_* columns follow the label-column order behind true_label, so argmax over them is the
        # predicted class index.
        for c in pred_df.columns:
            df_donor[f"prob_{c}"] = pred_df[c].values
        all_donors.append(df_donor)


df_res = pd.DataFrame(results)
df_res.to_csv(f"{save_dir}/results_summary.csv", index=False)

print("\n==== FINAL RESULTS (Baseline) ====")
print(df_res.groupby("fold")[["AUROC", "MacroF1"]].mean())
print("Mean AUROC:", df_res["AUROC"].mean())
print("Mean MacroF1:", df_res["MacroF1"].mean())


df_all = pd.concat(all_donors, ignore_index=True)
prob_cols = [c for c in df_all.columns if c.startswith("prob_")]

bin_stats = []
for cell_bin, g in df_all.groupby("cell_bin"):
    if len(np.unique(g["true_label"])) < 2:
        continue

    # Every (seed, fold, donor) prediction for this bin, pooled over seeds.
    y_true = g["true_label"].values
    y_prob = g[prob_cols].values

    # Reindex to 0..K-1 so a class missing from this bin becomes an all-zero column (skipped by
    # macro_auroc) and the columns line up with y_prob.
    true_df = pd.get_dummies(y_true).reindex(columns=range(y_prob.shape[1]), fill_value=0)
    pred_df = pd.DataFrame(y_prob, columns=true_df.columns)

    auroc_bin = macro_auroc(true_df, pred_df)
    f1_bin = f1_score(y_true, np.argmax(y_prob, axis=1), average="macro")

    bin_stats.append(
        {
            "cell_bin": cell_bin,
            "AUROC": auroc_bin,
            "MacroF1": f1_bin,
            "n_donors": len(g),
        }
    )

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
plt.title("Baseline AUROC vs Cell Bin (Cardio)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{save_dir}/AUROC_per_bin.pdf")
plt.show()

plt.figure(figsize=(7, 5))
sns.barplot(data=bin_df, x="cell_bin", y="MacroF1", color="salmon")
plt.title("Baseline MacroF1 vs Cell Bin (Cardio)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{save_dir}/F1_per_bin.pdf")
plt.show()
