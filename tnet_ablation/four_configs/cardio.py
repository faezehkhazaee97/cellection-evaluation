"""Same four-way T-Net sweep as covid.py, on Cardio with one bag per donor (Figure S2).

The label is the three-class obs `disease` (DCM / HCM / NF); bags are obs `donor_id` and folds come from
$DATA_DIR/cardio/cardio_5folds.csv. Because the task is multiclass, each fold records only macro
one-vs-rest AUROC, MacroAUPRC and MacroF1 (plus seed, fold, tnet_config), and no curves or confusion
matrices are drawn. per_donor_predictions.csv gets one prob_class_<i> column per class, in pd.get_dummies
order of `disease`. Everything goes to $RESULTS_DIR/R1-q2/Results/TnetConfigs/Cardio/<config>/.
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
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

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

    AUROC is NaN when roc_auc_score raises ValueError, e.g. when a class is missing from the fold.
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    num_classes = y_pred_proba.shape[1]

    out = {}
    try:
        out["AUROC"] = roc_auc_score(
            y_true, y_pred_proba, multi_class="ovr", average="macro"
        )
    except ValueError:
        out["AUROC"] = np.nan

    y_true_oh = np.eye(num_classes)[y_true]
    out["MacroAUPRC"] = compute_macro_auprc(y_true_oh, y_pred_proba)
    out["MacroF1"] = f1_score(y_true, y_pred, average="macro")

    return out


DATA_PATH = f"{DATA_DIR}/cardio/cardio_processed.h5ad"
FOLDS_PATH = f"{DATA_DIR}/cardio/cardio_5folds.csv"
SAVE_ROOT = f"{RESULTS_DIR}/R1-q2/Results/TnetConfigs/Cardio"
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

adata_full = adata.copy()
sample_key = "donor_id"
task_key = "disease"

for cfg in TNET_CONFIGS:
    print(f"\n===== CONFIG: {cfg['name']} =====\n")
    save_dir = f"{SAVE_ROOT}/{cfg['name']}"
    os.makedirs(save_dir, exist_ok=True)

    results = []
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

            y_true = np.argmax(true_df.values, axis=1)
            y_pred_proba = pred_df.values

            prob_df = pd.DataFrame(
                y_pred_proba,
                index=pred_df.index,
                columns=[f"prob_class_{i}" for i in range(y_pred_proba.shape[1])],
            )
            out_df = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "config": cfg["name"],
                            "seed": seed,
                            "fold": fold,
                            "donor": pred_df.index,
                            "true": y_true,
                        },
                        index=pred_df.index,
                    ),
                    prob_df,
                ],
                axis=1,
            )
            all_predictions.append(out_df)

            m = compute_metrics_multiclass(y_true, y_pred_proba)
            m["seed"] = seed
            m["fold"] = fold
            m["tnet_config"] = cfg["name"]
            results.append(m)

    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{save_dir}/metrics_per_fold.csv", index=False)
    df_res[["seed", "fold", "MacroAUPRC"]].to_csv(
        f"{save_dir}/MacroAUPRC_values.csv", index=False
    )

    if all_predictions:
        df_pred = pd.concat(all_predictions, ignore_index=True)
        df_pred.to_csv(f"{save_dir}/per_donor_predictions.csv", index=False)

    if not df_res.empty:
        print("\n==== MEAN METRICS ====")
        print(df_res.groupby("fold")[["AUROC", "MacroAUPRC", "MacroF1"]].mean())
        print("Overall mean AUROC:", df_res["AUROC"].mean())
