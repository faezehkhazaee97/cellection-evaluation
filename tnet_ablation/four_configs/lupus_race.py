"""Ancestry rather than disease: lupus donors, Asian vs White (obs `pop_cov`), four T-Net configs.

Otherwise lupus_disease.py: one bag per `ind_cov` donor, binary metrics and plots as in covid.py. Folds
come from $DATA_DIR/lupus/race_5folds.csv, and only the donors listed there are trained on or tested.
Results go to $RESULTS_DIR/R1-q2/Results/TnetConfigs/Lupus_Race/<config>/ (Figure S2).
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
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

import cellection

# P(class 1) at or above this is called class 1.
DECISION_THRESHOLD = 0.5
# Common x grid on [0, 1] for averaging ROC/PR curves across folds.
CURVE_GRID_POINTS = 200


def compute_macro_auprc(y_true_oh, y_pred_proba):
    """Mean one-vs-rest PR AUC over classes (trapezoidal, not average precision); both inputs are (N, C)."""
    n_classes = y_true_oh.shape[1]
    scores = []
    for c in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, c], y_pred_proba[:, c])
        scores.append(auc(recall, precision))
    return np.mean(scores)


def compute_metrics_binary(y_true, y_prob):
    """Metrics for one held-out fold, given y_prob = P(class 1).

    F1 and conf_mat threshold y_prob at DECISION_THRESHOLD. AUPRC is average precision with class 1
    positive; F1_case uses class 1 as positive, F1_control class 0.
    """
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

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


def plot_mean_curve(curves, metric, save_path):
    """Save the mean +/- SD of per-fold curves, each linearly interpolated onto CURVE_GRID_POINTS x values.

    curves holds dicts with fpr/tpr when metric == "ROC", recall/precision otherwise.
    """
    plt.figure(figsize=(6, 5))

    if metric == "ROC":
        xs = [c["fpr"] for c in curves]
        ys = [c["tpr"] for c in curves]
        xlabel, ylabel = "False Positive Rate", "True Positive Rate"
    else:
        xs = [c["recall"] for c in curves]
        ys = [c["precision"] for c in curves]
        xlabel, ylabel = "Recall", "Precision"

    grid = np.linspace(0, 1, CURVE_GRID_POINTS)
    ys_interp = [np.interp(grid, xs[i], ys[i]) for i in range(len(xs))]
    mean_y = np.mean(ys_interp, axis=0)
    std_y = np.std(ys_interp, axis=0)

    plt.plot(grid, mean_y, label=f"Mean {metric}")
    plt.fill_between(grid, mean_y - std_y, mean_y + std_y, alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Mean {metric} Curve +/- SD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


DATA_PATH = f"{DATA_DIR}/lupus/lupus_processed.h5ad"
FOLDS_PATH = f"{DATA_DIR}/lupus/race_5folds.csv"
SAVE_ROOT = f"{RESULTS_DIR}/R1-q2/Results/TnetConfigs/Lupus_Race"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use_tnet1/use_tnet2 = None inherit use_tnet, so the first two rows switch both T-Nets together.
TNET_CONFIGS = [
    {"name": "tnet_both_off", "use_tnet": False, "use_tnet1": None, "use_tnet2": None},
    {"name": "tnet_both_on", "use_tnet": True, "use_tnet1": None, "use_tnet2": None},
    {"name": "tnet1_on_tnet2_off", "use_tnet": False, "use_tnet1": True, "use_tnet2": False},
    {"name": "tnet1_off_tnet2_on", "use_tnet": False, "use_tnet1": False, "use_tnet2": True},
]

adata = sc.read_h5ad(DATA_PATH)
folds = pd.read_csv(FOLDS_PATH, index_col=0)
num_folds = folds["fold"].nunique()

sample_key = "ind_cov"
task_key = "pop_cov"
adata_full = adata.copy()

for cfg in TNET_CONFIGS:
    print(f"\n===== CONFIG: {cfg['name']} =====\n")
    save_dir = f"{SAVE_ROOT}/{cfg['name']}"
    os.makedirs(save_dir, exist_ok=True)

    results = []
    roc_curves = []
    pr_curves = []
    conf_mats = []
    all_predictions = []

    for seed in range(3):
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n=== SEED {seed} ===")
        for fold in range(1, num_folds + 1):
            print(f"\n--- Fold {fold} ---")

            train_donors = folds.loc[folds["fold"] != fold, "samples"]
            test_donors = folds.loc[folds["fold"] == fold, "samples"]

            adata_train = adata_full[adata_full.obs[sample_key].isin(train_donors)]
            adata_test = adata_full[adata_full.obs[sample_key].isin(test_donors)].copy()

            # Skip degenerate folds (two or fewer training donors).
            if adata_train.obs[sample_key].nunique() <= 2:
                continue

            obj = cellection.cellectiion_object(
                adata=adata_train,
                task_type="classification",
                task_key=task_key,
                sample_key=sample_key,
                batch_key=None,
                input_type="measurement",
                sparse_input=True,
                InstanceEncoder=True,
                use_tnet=cfg["use_tnet"],
                use_tnet1=cfg["use_tnet1"],
                use_tnet2=cfg["use_tnet2"],
                val_size=0.2,
                aggregator="gated_attention",
                global_features=256,
                attention_dim=64,
                max_epochs=200,
                learning_rate=1e-4,
                batch_size=15,
                seed=seed,
                device=DEVICE,
                hidden_layer=[256, 32],
                activation=nn.ReLU(),
                layernorm=True,
                dropout_rate=0.1,
                patience=20,
                save_model=True,
                save_path=f"{save_dir}/seed_{seed}/fold_{fold}",
            )

            obj.prepare()
            obj.train()

            obj.adata = adata_test
            obj.prepare_inference()

            _, true_df, pred_df, _, _, _ = obj.inference()

            # Columns are in pd.get_dummies order of the label; column 1 is scored as positive.
            y_true = np.argmax(true_df.values, axis=1)
            y_prob = pred_df.values[:, 1]

            all_predictions.append(
                pd.DataFrame(
                    {
                        "config": cfg["name"],
                        "seed": seed,
                        "fold": fold,
                        "donor": pred_df.index,
                        "true": y_true,
                        "prob": y_prob,
                    }
                )
            )

            m = compute_metrics_binary(y_true, y_prob)
            m["seed"] = seed
            m["fold"] = fold
            m["tnet_config"] = cfg["name"]
            results.append(m)

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_curves.append({"fpr": fpr, "tpr": tpr})

            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_curves.append({"precision": precision, "recall": recall})

            cm = m["conf_mat"].astype(float)
            cm_percent = cm / cm.sum(axis=1, keepdims=True)
            conf_mats.append(cm_percent)

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{save_dir}/metrics_per_fold.csv", index=False)
    df_res[["seed", "fold", "MacroAUPRC"]].to_csv(
        f"{save_dir}/MacroAUPRC_values.csv", index=False
    )

    conf_mats = [cm for cm in conf_mats if cm.shape == (2, 2)]
    if conf_mats:
        np.save(f"{save_dir}/conf_mats.npy", np.stack(conf_mats))
        mean_cm = np.mean(conf_mats, axis=0)
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            mean_cm * 100,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            xticklabels=["WHITE", "ASIAN"],
            yticklabels=["WHITE", "ASIAN"],
        )
        plt.title("Mean Confusion Matrix (% per row)")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix_mean.pdf")
        plt.close()

    if roc_curves:
        plot_mean_curve(roc_curves, "ROC", f"{save_dir}/mean_ROC_curve.pdf")
    if pr_curves:
        plot_mean_curve(pr_curves, "PR", f"{save_dir}/mean_PR_curve.pdf")

    if all_predictions:
        df_pred = pd.concat(all_predictions, ignore_index=True)
        df_pred.to_csv(f"{save_dir}/per_donor_predictions.csv", index=False)

    if not df_res.empty:
        print("\n==== MEAN METRICS ====")
        print(df_res.groupby("fold")[["AUROC", "AUPRC", "MacroAUPRC", "MacroF1"]].mean())
        print("Overall mean AUROC:", df_res["AUROC"].mean())
