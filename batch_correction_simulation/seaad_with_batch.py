"""SEA-AD MTG glia, AD vs control, with donor sex as CELLECTION's batch covariate.

Paired with seaad_without_batch.py, which uses the same donor splits without a covariate;
check_batch_balance.py tabulates sex against disease. Cells without obs["Supertype"] are
dropped; donors (obs["donor_id"]) are the bags and obs["disease"] the label. For seeds 10-14,
donors are shuffled after np.random.seed(seed) and split 80/20 into training and held-out
donors. CELLECTION (instance encoder [256, 32], gated attention, up to 200 epochs, patience 20)
keeps val_size=0.2 of the training donors for early stopping; AUROC and macro-F1 are computed
on the held-out donors.

Reads $DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad. Writes seed_<s>/ model folders and
metrics_with_batch_5seeds.csv (seed, setting, AUROC, MacroF1) to
$RESULTS_DIR/R1-q1/Results/withBatch/, and prints the mean and SD over seeds. Uses a GPU when
available.
"""

# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import scanpy as sc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score

import cellection

DATA_PATH = f"{DATA_DIR}/sea-ad/MTG_glia_intersection_norm.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R1-q1/Results/withBatch"
os.makedirs(SAVE_DIR, exist_ok=True)
USE_BATCH=True
sample_key = "donor_id"
task_key = "disease"        # AD vs control
celltype_key = "Supertype"  # fine glial types
BATCH_KEY="sex"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DONOR_FRACTION = 0.8  # share of donors used for training; the rest are held out
DECISION_THRESHOLD = 0.5  # probability cut-off for the class in column 1 of the predictions

adata = sc.read_h5ad(DATA_PATH)

# null Supertype = not an annotated glial cell
adata = adata[adata.obs[celltype_key].notna()].copy()

print("Donors:", adata.obs[sample_key].nunique())
print("Cells:", adata.n_obs)
print(adata.obs.columns.tolist())
SEEDS = [10, 11, 12, 13, 14]
all_metrics = []
for seed in SEEDS:
    print(f"\n================ SEED {seed} ================\n")

    # Split by donor so no donor has cells on both sides.
    donors = np.array(adata.obs[sample_key].unique())
    np.random.seed(seed)
    np.random.shuffle(donors)

    n_train = int(TRAIN_DONOR_FRACTION * len(donors))
    train_donors = donors[:n_train]
    val_donors   = donors[n_train:]

    adata_train = adata[adata.obs[sample_key].isin(train_donors)].copy()
    adata_val   = adata[adata.obs[sample_key].isin(val_donors)].copy()

    print("Train donors:", adata_train.obs[sample_key].nunique())
    print("Val donors:", adata_val.obs[sample_key].nunique())
    print("Train assay counts:\n", adata_train.obs[BATCH_KEY].value_counts())
    print("Val assay counts:\n", adata_val.obs[BATCH_KEY].value_counts())

    obj = cellection.cellectiion_object(
        adata=adata_train,
        task_type="classification",
        task_key=task_key,
        sample_key=sample_key,
        batch_key=BATCH_KEY,
        model_type="classification",
        input_type="measurement",
        sparse_input=True,
        InstanceEncoder=True,
        val_size=0.2,
        aggregator="gated_attention",
        global_features=256,
        attention_dim=64,
        hidden_layer=[256, 32],
        activation=nn.ReLU(),
        dropout_rate=0.1,
        learning_rate=1e-4,
        batch_size=15,
        max_epochs=200,
        patience=20,
        device=DEVICE,
        seed=seed,
        save_model=True,
        save_path=f"{SAVE_DIR}/seed_{seed}"
    )

    obj.prepare()
    obj.train()

    obj.adata = adata_val
    obj.prepare_inference()

    (
        sample_metadata,
        true_label_df,
        pred_label_df,
        embedding_df,
        global_feature_df,
        attention_mtx_raw_df
    ) = obj.inference()

    y_true = np.argmax(true_label_df.values, axis=1)
    y_prob = pred_label_df.values[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    auroc = roc_auc_score(y_true, y_prob)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"[Seed {seed}] AUROC = {auroc:.4f}, Macro-F1 = {macro_f1:.4f}")

    all_metrics.append({
        "seed": seed,
        "setting": "with_batch" if USE_BATCH else "no_batch",
        "AUROC": auroc,
        "MacroF1": macro_f1
    })
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(
    f"{SAVE_DIR}/metrics_{'with_batch' if USE_BATCH else 'no_batch'}_5seeds.csv",
    index=False
)

print("\n===== SUMMARY =====")
print(metrics_df.groupby("setting")[["AUROC", "MacroF1"]].agg(["mean", "std"]))