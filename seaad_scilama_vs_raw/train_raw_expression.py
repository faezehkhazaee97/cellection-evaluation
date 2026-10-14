"""SEA-AD MTG glia, AD vs control from raw expression: the baseline arm of the sciLaMA comparison.

CELLECTION classifies donors as AD vs control (task_key="disease", one bag per donor_id) from
normalized expression (input_type="measurement"): instance encoder [256, 32] with layer norm and
dropout 0.1, T-Nets, gated-attention pooling, lr 1e-4, 15 bags per batch, at most 200 epochs,
patience 20.

For each seed in SEEDS (17, 16, 9, 18-22) the donors are split 80/20 by a stratified
train_test_split(random_state=SEED). The split is saved to seaad_splits/ on first use and re-read
afterwards, which is how the sciLaMA arm gets the same donors. cellection holds out another 20% of
the training donors for early stopping. Test donors are scored with obj.inference() (best
checkpoint), and the metrics use the predicted probability of the second disease category in
pd.get_dummies order, with F1 and the confusion matrix at 0.5.

Reads $DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad (obs donor_id, disease). Writes to
$RESULTS_DIR/R1-q3/Results/measurement_baseline_NoFolds/:
  metrics_5seed.csv      one row per seed (all of SEEDS, despite the name): AUROC, AUPRC,
                         MacroAUPRC, MacroF1, F1_case, F1_control, conf_mat, seed
  predictions_5seed.csv  seed, donor, true, prob for every test donor and seed
  seed_<SEED>/           model checkpoints
and, when missing, $RESULTS_DIR/R1-q3/Results/seaad_splits/disease_seed_<SEED>_split.csv
(donor_id, split).
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_recall_curve, auc
)
from sklearn.model_selection import train_test_split

import cellection


def compute_macro_auprc(y_true_oh, y_pred_proba):
    """Mean over classes of the one-vs-rest precision-recall AUC (trapezoidal `auc`)."""
    n_classes = y_true_oh.shape[1]
    scores = []
    for c in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, c], y_pred_proba[:, c])
        scores.append(auc(recall, precision))
    return np.mean(scores)


def compute_metrics_binary(y_true, y_prob):
    """Metrics from 0/1 labels and the class-1 score; F1 and confusion matrix use y_prob >= 0.5."""
    y_pred = (y_prob >= 0.5).astype(int)

    out = {}
    out["AUROC"] = roc_auc_score(y_true, y_prob)
    out["AUPRC"] = average_precision_score(y_true, y_prob)

    y_true_oh = np.eye(2)[y_true]
    y_pred_proba = np.column_stack([1 - y_prob, y_prob])
    out["MacroAUPRC"] = compute_macro_auprc(y_true_oh, y_pred_proba)

    out["MacroF1"] = f1_score(y_true, y_pred, average="macro")
    out["F1_case"] = f1_score(y_true, y_pred, pos_label=1)
    out["F1_control"] = f1_score(y_true, y_pred, pos_label=0)
    out["conf_mat"] = confusion_matrix(y_true, y_pred)

    return out


adata = sc.read_h5ad(
    f"{DATA_DIR}/sea-ad/MTG_glia_intersection_norm.h5ad"
)
sample_key = "donor_id"
task_key = "disease"
save_dir = f"{RESULTS_DIR}/R1-q3/Results/measurement_baseline_NoFolds"
os.makedirs(save_dir, exist_ok=True)
split_dir = f"{RESULTS_DIR}/R1-q3/Results/seaad_splits"
os.makedirs(split_dir, exist_ok=True)
SEEDS = [17,16,9,18, 19,20,21,22]
results = []
all_predictions = []
for SEED in SEEDS:
    print("\n==============================")
    print(f"        SEED {SEED}")
    print("==============================")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    donor_df = (
        adata.obs[[sample_key, task_key]]
        .drop_duplicates()
        .set_index(sample_key)
    )

    donors = donor_df.index.values
    donor_labels = donor_df[task_key].values

    split_path = os.path.join(split_dir, f"{task_key}_seed_{SEED}_split.csv")
    if os.path.exists(split_path):
        split_df = pd.read_csv(split_path)
        train_donors = split_df.loc[split_df["split"] == "train", sample_key].values
        test_donors = split_df.loc[split_df["split"] == "test", sample_key].values
    else:
        train_donors, test_donors = train_test_split(
            donors,
            test_size=0.20,
            random_state=SEED,
            stratify=donor_labels,
        )
        split_df = pd.DataFrame(
            {
                sample_key: np.concatenate([train_donors, test_donors]),
                "split": ["train"] * len(train_donors) + ["test"] * len(test_donors),
            }
        )
        split_df.to_csv(split_path, index=False)

    adata_train = adata[adata.obs[sample_key].isin(train_donors)].copy()
    adata_test = adata[adata.obs[sample_key].isin(test_donors)].copy()

    print("Train donors:", len(train_donors))
    print("Test donors:", len(test_donors))
    print("Train cells:", adata_train.n_obs)
    print("Test cells:", adata_test.n_obs)
    obj = cellection.cellectiion_object(
        adata=adata_train,
        task_type="classification",
        task_key=task_key,
        sample_key=sample_key,
        batch_key=None,
        input_type="measurement",
        sparse_input=True,
        InstanceEncoder=True,
        use_tnet=True,
        val_size=0.2,
        aggregator="gated_attention",
        global_features=256,
        attention_dim=64,
        max_epochs=200,
        learning_rate=1e-4,
        batch_size=15,
        seed=SEED,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hidden_layer=[256, 32],
        activation=nn.ReLU(),
        layernorm=True,
        dropout_rate=0.1,
        patience=20,
        save_model=True,
        save_path=f"{save_dir}/seed_{SEED}"
    )
    obj.prepare()
    obj.train()
    obj.adata = adata_test
    obj.prepare_inference()
    _, true_df, pred_df, _, _, _ = obj.inference()

    # y_true: index of the true disease column; y_prob: predicted probability of the second column
    # (pd.get_dummies order).
    y_true = np.argmax(true_df.values, axis=1)
    y_prob = pred_df.values[:, 1]

    m = compute_metrics_binary(y_true, y_prob)
    m["seed"] = SEED
    results.append(m)

    all_predictions.append(pd.DataFrame({
        "seed": SEED,
        "donor": pred_df.index,
        "true": y_true,
        "prob": y_prob
    }))

df = pd.DataFrame(results)
df.to_csv(f"{save_dir}/metrics_5seed.csv", index=False)

df_pred = pd.concat(all_predictions, ignore_index=True)
df_pred.to_csv(f"{save_dir}/predictions_5seed.csv", index=False)

print("\n===== FINAL 5-SEED RESULTS =====")
print(df[["seed", "AUROC", "AUPRC", "MacroAUPRC", "MacroF1"]].mean())
