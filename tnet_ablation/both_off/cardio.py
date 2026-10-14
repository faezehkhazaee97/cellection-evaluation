"""Cardio by donor (three-class `disease`, bags = `donor_id`) with both T-Nets off.

Mirror of ../both_on/cardio.py with use_tnet=False: same folds (cardio_5folds.csv), multiclass metrics
and prediction files, written to $RESULTS_DIR/R1-q2/Results/WithoutTnet/Cardio/.
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
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

import cellection


def compute_macro_auprc(y_true_oh, y_pred_proba):
    """Mean one-vs-rest PR AUC over classes (trapezoidal, not average precision); both inputs are (N, C)."""
    n_classes = y_true_oh.shape[1]
    scores = []
    for c in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, c], y_pred_proba[:, c])
        scores.append(auc(recall, precision))
    return np.mean(scores)


def compute_metrics_multiclass(y_true, y_pred_proba):
    """Macro one-vs-rest AUROC, MacroAUPRC and argmax MacroF1 for one held-out fold.

    AUROC is NaN if roc_auc_score raises (any exception), e.g. when a class is missing from the fold.
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    C = y_pred_proba.shape[1]

    out = {}

    try:
        out["AUROC"] = roc_auc_score(
            y_true, y_pred_proba, multi_class="ovr", average="macro"
        )
    except:
        out["AUROC"] = np.nan

    y_true_oh = np.eye(C)[y_true]
    out["MacroAUPRC"] = compute_macro_auprc(y_true_oh, y_pred_proba)

    out["MacroF1"] = f1_score(y_true, y_pred, average="macro")

    return out


print(torch.cuda.is_available())

adata = sc.read_h5ad(
    f"{DATA_DIR}/cardio/cardio_processed.h5ad"
)
folds = pd.read_csv(
    f"{DATA_DIR}/cardio/cardio_5folds.csv",
    index_col=0,
)


num_folds = folds["fold"].nunique()

save_dir = (
    f"{RESULTS_DIR}/R1-q2/Results/WithoutTnet/Cardio"
)
os.makedirs(save_dir, exist_ok=True)

adata.obs.biosample_id.value_counts()
adata.obs.donor_id.value_counts()
adata.obs[['disease','donor_id']].value_counts().sort_index()
adata.obs['cell_type_leiden0.6'].value_counts()
adata_full = adata.copy()

class_num = 3
sample_key = 'donor_id'
task_key = 'disease'
task_name = 'cardio_cls'
ct_key = 'cell_type_leiden0.6'
num_folds = folds["fold"].nunique()
results = []
roc_curves = []
pr_curves = []
conf_mats = []

all_predictions = []

for SEED in range(5):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n=== SEED {SEED} ===")
    for fold in range(1, num_folds+1):

        print(f"\n--- Fold {fold} ---")

        train_donors = folds.loc[folds["fold"] != fold, "samples"]
        test_donors  = folds.loc[folds["fold"] == fold, "samples"]

        adata_train = adata_full[adata_full.obs[sample_key].isin(train_donors)]
        print("Unique labels in train:", adata_train.obs[task_key].unique())
        print("Value counts:", adata_train.obs[task_key].value_counts())

        adata_test  = adata_full[adata_full.obs[sample_key].isin(test_donors)].copy()

        # Skip degenerate folds (two or fewer training donors).
        if adata_train.obs[sample_key].nunique() <= 2:
            continue
        print("Train genes:", adata_train.var_names.shape)
        print("Test genes:",  adata_test.var_names.shape)
        obj = cellection.cellectiion_object(
            adata=adata_train,
            task_type="classification",
            task_key=task_key,
            sample_key=sample_key,
            batch_key=None,
            input_type="measurement",
            sparse_input=True,
            InstanceEncoder=True,
            use_tnet=False,
            val_size=0.2,
            aggregator="gated_attention",
            global_features=256,
            attention_dim=64,
            max_epochs=200,
            learning_rate=1e-4,
            batch_size=15,
            seed=SEED,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            hidden_layer=[256,32],
            activation=nn.ReLU(),
            layernorm=True,
            dropout_rate=0.1,
            patience=20,
            save_model=True,
            save_path=f"{save_dir}/seed_{SEED}/fold_{fold}",
        )

        # prepare() builds train_loader, which the checks below iterate over.
        obj.prepare()

        # Loader sanity checks: bag shapes of the first 3 batches, labels of the first 6.
        print("\n=== TRAIN loader shapes AFTER FIX ===")
        for i, (bags, _, _, _) in enumerate(obj.train_loader):
            print("Train batch", i, "shape:", bags.shape)
            if i == 2:
                break

        print("\n=== CHECKING LABELS INSIDE TRAIN LOADER ===")
        for i, (_, labels, _, _) in enumerate(obj.train_loader):
            print(f"Batch {i} unique labels:", torch.unique(labels.cpu()).tolist())
            if i == 5:
                break

        obj.train()

        # Reuse the trained object to predict the held-out donors.
        obj.adata = adata_test
        obj.prepare_inference()

        _, true_df, pred_df, _, _, _ = obj.inference()

        y_true = np.argmax(true_df.values, axis=1)
        y_pred_proba = pred_df.values

        prob_df = pd.DataFrame(
            y_pred_proba,
            index=pred_df.index,
            columns=[f"prob_class_{i}" for i in range(y_pred_proba.shape[1])]
        )

        out_df = pd.concat([
            pd.DataFrame({
                "seed": SEED,
                "fold": fold,
                "donor": pred_df.index,
                "true": y_true
            }, index=pred_df.index),
            prob_df
        ], axis=1)

        all_predictions.append(out_df)

        m = compute_metrics_multiclass(y_true, y_pred_proba)
        m["seed"] = SEED
        m["fold"] = fold
        results.append(m)


df_res = pd.DataFrame(results)
df_res[["seed","fold","MacroAUPRC"]].to_csv(f"{save_dir}/MacroAUPRC_values.csv", index=False)
print("Mean MacroAUPRC:", df_res["MacroAUPRC"].mean())

df_res.to_csv(f"{save_dir}/metrics_per_fold.csv", index=False)

print("\n==== MEAN METRICS ====")
print(df_res.groupby("fold")[["AUROC","MacroAUPRC","MacroF1"]].mean())
print("Overall mean AUROC:", df_res["AUROC"].mean())


df_pred = pd.concat(all_predictions, ignore_index=True)
df_pred.to_csv(f"{save_dir}/per_donor_predictions.csv", index=False)
