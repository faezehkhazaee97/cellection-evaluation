"""Ordinal CELLECTION on Haniffa COVID-19 severity with learned cut points (Figure S10).

This is the learned half of the cut-point comparison; fixed_cutpoints.py is the other half.
Worst_Clinical_Status is mapped to eight ordered levels (Healthy < Non-covid < Asymptomatic < Mild <
Moderate < Severe < Critical < Death) and dense-ranked to 0..K-1 in obs["severity_rank"]; bags are
obs["sample_id"]. The network outputs one scalar score f per bag, and the ordinal loss has K-1 cut
points c_k with P(y <= k) = sigmoid(c_k - f).

For each starting range in CUTPOINT_INIT_EXPERIMENTS ([-5, 5] and [-10, 10]) and each seed, the cut
points are reset after prepare() to K-1 evenly spaced values over that range and then trained along
with the network (no instance encoder; T-Net and gated attention; cellection's 80/20 patient split,
whose validation part also drives early stopping). Validation patients are scored on CPU with the
best checkpoint; bags with fewer than 2 cells are skipped. Metrics: Spearman and Pearson between the
true rank and (a) the raw score ("*_logits") or (b) the decoded class, i.e. the number of cut points
below the score ("*_decoded"), plus the accuracy of the decoded class.

Reads $DATA_DIR/haniffa21_intersection.h5ad. Writes to $RESULTS_DIR/R1-Q11/Results/with_new/, with
<exp> = trained_init_-5_5 or trained_init_-10_10:
  haniffa_cellection_metrics_cutpoints_True_<exp>.csv   one row per seed: seed, train_cutpoints,
      cutpoint_init_range, experiment, spearman/pearson (+ p) for *_logits and *_decoded,
      accuracy_decoded
  haniffa_cellection_metrics_cutpoints_True_all_experiments.csv   the rows of both <exp> together
  haniffa_cellection_seed<S>_cutpoints_True_<exp>_{true,pred}_labels.csv   per validation patient:
      true rank (severity_rank) and decoded class (predicted_ordinal_score)
  haniffa_severity_scatter_seed<S>_cutpoints_True_<exp>.pdf   raw score by true severity level
  checkpoints_<exp>/severity_rank_ordinal_SEED<S>/   checkpoints
With TRAIN_CUTPOINTS = False the reset cut points stay frozen and the file names say
"cutpoints_False"; the fixed-cut-point results come from fixed_cutpoints.py instead.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
# The env var RESULTS_DIR goes into RESULTS_ROOT; RESULTS_DIR below is this script's output folder.
RESULTS_ROOT = os.environ.get("RESULTS_DIR", "results")  # output root


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import spearmanr

import cellection

RESULTS_DIR = f"{RESULTS_ROOT}/R1-Q11/Results/with_new"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_PATH = f"{DATA_DIR}/haniffa21_intersection.h5ad"
SEEDS = [7,8,9,10,6, 5, 2]

SAMPLE_KEY = "sample_id"
TASK_KEY = "severity_rank"

TRAIN_CUTPOINTS = True   # False keeps the cut points fixed at their initial values
CUTPOINT_INIT_EXPERIMENTS = [
    {"name": "trained_init_-5_5", "range": (-5.0, 5.0)},
    {"name": "trained_init_-10_10", "range": (-10.0, 10.0)},
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adata_base = sc.read_h5ad(DATA_PATH)


def decode_ordinal_class(logit, cutpoints):
    """Ordinal class of a scalar score: the number of cut points strictly below it."""
    return int(torch.sum(logit > cutpoints).item())


adata_base.obs["Worst_Clinical_Status"] = (
    adata_base.obs["Worst_Clinical_Status"]
    .astype(str)
    .str.strip()
)

adata_base.obs["severity_order"] = adata_base.obs["Worst_Clinical_Status"].map({
    "Healthy": 0,
    "Non-covid": 1,
    "Asymptomatic": 2,
    "Mild": 3,
    "Moderate": 4,
    "Severe": 5,
    "Critical": 6,
    "Death": 7
})

# cellection wants ordinal labels 0..K-1. The dense rank keeps the order above and closes the gap
# if a level is absent from the data.
adata_base.obs[TASK_KEY] = (
    adata_base.obs["severity_order"]
    .rank(method="dense")
    .astype(int) - 1
)

print("Severity ranks:")
print(adata_base.obs[TASK_KEY].value_counts().sort_index())

metrics = []
for exp in CUTPOINT_INIT_EXPERIMENTS:
    exp_name = exp["name"]
    cutpoint_range = exp["range"]
    exp_metrics = []
    for SEED in SEEDS:

        print("\n" + "="*60)
        print(
            f"Running SEED {SEED} | train_cutpoints={TRAIN_CUTPOINTS} | exp={exp_name}"
        )
        print("="*60)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        adata_run = adata_base.copy()
        obj = cellection.cellectiion_object(
            adata=adata_run,
            task_type="ordinal",
            task_key=TASK_KEY,
            sample_key=SAMPLE_KEY,
            batch_key=None,

            model_type="ordinal",
            input_type="measurement",
            sparse_input=True,

            InstanceEncoder=False,
            hidden_layer=[256, 32],
            use_tnet=True,

            aggregator="gated_attention",
            global_features=256,
            attention_dim=64,

            train_cutpoints=TRAIN_CUTPOINTS,

            learning_rate=1e-4,
            batch_size=15,
            val_size=0.2,
            max_epochs=200,
            patience=20,

            seed=SEED,
            device=DEVICE,

            save_model=True,
            save_path=os.path.join(RESULTS_DIR, f"checkpoints_{exp_name}")  # one folder per initial range
        )

        obj.prepare()
        # Reset the package's initial cut points to K-1 evenly spaced values over cutpoint_range.
        # Assigning .data keeps the same nn.Parameter, so the optimiser built in prepare() still
        # updates it when train_cutpoints=True.
        if hasattr(obj, "criterion") and hasattr(obj.criterion, "cutpoints"):
            low, high = cutpoint_range
            count = obj.criterion.cutpoints.numel()
            fixed = torch.linspace(low, high, steps=count)
            obj.criterion.cutpoints.data = fixed.to(obj.criterion.cutpoints.device)
            cutpoints_init = obj.criterion.cutpoints.detach().cpu().clone()
            print("Initial cutpoints:", cutpoints_init)
        obj.train()
        if hasattr(obj, "criterion") and hasattr(obj.criterion, "cutpoints"):
            cutpoints_final = obj.criterion.cutpoints.detach().cpu().clone()
            print("Final cutpoints:", cutpoints_final)

        # Score the validation patients on CPU with the best checkpoint.
        obj.pn_checkpoint = obj.pn_checkpoint.to("cpu")
        obj.pn_checkpoint.eval()

        if obj.enc_checkpoint is not None:
            obj.enc_checkpoint = obj.enc_checkpoint.to("cpu")
            obj.enc_checkpoint.eval()

        val_dataset = obj.val_dataset  # the 20% of patients cellection held out

        y_true = []
        y_pred = []
        y_score = []

        with torch.no_grad():
            for i in range(len(val_dataset)):
                # One validation patient: zero-padded bag (max_cells, genes), integer rank label,
                # true number of cells, cell ids.
                bag, label, length, meta_id = val_dataset[i]

                if length <= 1:
                    continue

                # Drop the padding; the network takes (batch, genes, cells).
                bag = bag[:length].unsqueeze(0).permute(0, 2, 1).to("cpu")

                if obj.InstanceEncoder:
                    enc_out = obj.enc_checkpoint(
                        bag.squeeze(0).transpose(0, 1), None
                    ).transpose(0, 1).unsqueeze(0)
                    logits = obj.pn_checkpoint(enc_out)[0]
                else:
                    logits = obj.pn_checkpoint(bag)[0]

                if hasattr(obj.criterion, "cutpoints"):
                    cutpoints = obj.criterion.cutpoints.to(logits.device)
                    pred = decode_ordinal_class(logits.squeeze(), cutpoints)
                else:
                    # fallback; not reached in ordinal mode
                    pred = logits.item()
                y_true.append(label.item())
                y_pred.append(pred)
                y_score.append(logits.squeeze().item())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        from scipy.stats import spearmanr, pearsonr

        from sklearn.metrics import accuracy_score

        spearman_res = spearmanr(y_true, y_score)
        spearman = spearman_res.correlation
        spearman_p = spearman_res.pvalue
        pearson, pearson_p = pearsonr(y_true, y_score)
        spearman_decoded_res = spearmanr(y_true, y_pred)
        spearman_decoded = spearman_decoded_res.correlation
        spearman_decoded_p = spearman_decoded_res.pvalue
        pearson_decoded, pearson_decoded_p = pearsonr(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        print(f"Ordinal accuracy: {acc:.3f}")

        print(f"SEED {SEED}")
        print(f"  Spearman: {spearman:.4f} (p={spearman_p:.3g})")
        print(f"  Pearson : {pearson:.4f} (p={pearson_p:.3g})")
        print(f"  Spearman (decoded): {spearman_decoded:.4f} (p={spearman_decoded_p:.3g})")
        print(f"  Pearson  (decoded): {pearson_decoded:.4f} (p={pearson_decoded_p:.3g})")
        metrics_entry = {
            "seed": SEED,
            "train_cutpoints": TRAIN_CUTPOINTS,
            "cutpoint_init_range": cutpoint_range,
            "experiment": exp_name,
            "spearman_logits": spearman,
            "spearman_p_logits": spearman_p,
            "pearson_logits": pearson,
            "pearson_p_logits": pearson_p,
            "spearman_decoded": spearman_decoded,
            "spearman_p_decoded": spearman_decoded_p,
            "pearson_decoded": pearson_decoded,
            "pearson_p_decoded": pearson_decoded_p,
            "accuracy_decoded": acc
        }
        exp_metrics.append(metrics_entry)
        metrics.append(metrics_entry)

        severity_map = {
            0: "Healthy",
            1: "Non-covid",
            2: "Asymptomatic",
            3: "Mild",
            4: "Moderate",
            5: "Severe",
            6: "Critical",
            7: "Death",
        }

        plot_df = pd.DataFrame({
            "True Severity": y_true,
            "Predicted Severity": y_pred,
            "Predicted Score": y_score
        })

        severity_order = [
            "Healthy",
            "Non-covid",
            "Asymptomatic",
            "Mild",
            "Moderate",
            "Severe",
            "Critical",
            "Death",
        ]
        plot_df["Severity Label"] = pd.Categorical(
            plot_df["True Severity"].map(dict(enumerate(severity_order))),
            categories=severity_order,
            ordered=True
        )
        print(
            plot_df
            .groupby("Severity Label")["Predicted Severity"]
            .mean()
        )
        plt.figure(figsize=(6, 5))

        sns.stripplot(
            data=plot_df,
            x="Severity Label",
            y="Predicted Score",
            jitter=0.25,
            size=8,
            alpha=0.8,
            color="#6baed6"
        )

        plt.xlabel("Haniffa data severity labels")
        plt.ylabel("Predicted Score")
        plt.title(
            "Haniffa model → Haniffa test data\n"
            f"Spearman r={spearman:.2f}, p={spearman_p:.2g}",
            fontsize=10
        )
        plt.gca().set_yticks([])

        plt.xticks(rotation=45, ha="right")
        sns.despine()
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"haniffa_severity_scatter_seed{SEED}_cutpoints_{TRAIN_CUTPOINTS}_{exp_name}.pdf"
            ),
            dpi=300
        )
        plt.close()

        out_prefix = os.path.join(
            RESULTS_DIR,
            f"haniffa_cellection_seed{SEED}_cutpoints_{TRAIN_CUTPOINTS}_{exp_name}"
        )

        y_true_df = pd.DataFrame(y_true, columns=["severity_rank"])
        y_pred_df = pd.DataFrame(y_pred, columns=["predicted_ordinal_score"])

        y_true_df.to_csv(f"{out_prefix}_true_labels.csv", index=False)
        y_pred_df.to_csv(f"{out_prefix}_pred_labels.csv", index=False)

    exp_metrics_df = pd.DataFrame(exp_metrics)
    exp_metrics_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"haniffa_cellection_metrics_cutpoints_{TRAIN_CUTPOINTS}_{exp_name}.csv"
        ),
        index=False
    )
    print("\nSaved metrics:")
    print(exp_metrics_df)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        f"haniffa_cellection_metrics_cutpoints_{TRAIN_CUTPOINTS}_all_experiments.csv"
    ),
    index=False
)

print("\nSaved metrics summary:")
print(metrics_df)
print("\nDONE.")
