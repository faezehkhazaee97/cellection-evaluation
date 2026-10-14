"""How do CELLECTION's training time and GPU memory grow with patient count and cells per patient?

The data are the Ziegler COVID-19 donors (58 in total, about 46-47 per training fold). The grid crosses
  patients_grid = [14, 29, 58, 116] training bags drawn from the fold's training donors. The 58 and
    116 settings need replacement, and a donor drawn k times becomes k bags ("<id>", "<id>_dup1", ...).
  cells_grid = [119, 409, 815, 2140] cells per bag. Every training and test bag is resized to exactly
    this size: larger donors are subsampled, smaller ones padded with Gaussian-noise copies of their
    own cells (subsample_cells_per_patient).
so a run trains on 1,666 to 248,240 cells. For seeds 0-1, folds 1-5 of covid_5folds.csv and every grid
point (160 runs), CELLECTION (case vs control, up to 200 epochs, early-stopping patience 20) is trained
on a CUDA GPU, which is required. The wall-clock time and peak allocated GPU memory of obj.train() are
recorded, and the resized test donors of the fold are scored (AUROC, AUPRC, F1).

The label is obs `disease_id`, derived from `disease__ontology_label` when missing ("COVID-19" and
"long COVID-19" -> "case", anything else -> "control"). All metrics treat class index 1 as positive,
which with case/control labels in alphabetical order is "control", so F1_case actually holds the F1 of
"control" and F1_control that of "case".

Reads covid_Ziegler_processed.h5ad and covid_5folds.csv from $DATA_DIR/covid/. Writes to
$RESULTS_DIR/R1-Q13/Code/Results/Covid/:
  scaling_runtime_gpu.csv     one row per run: seed, fold, train_patients, cells_per_patient,
                              total_cells_train, gpu_memory_mb (in MiB), epochs_run, epoch_time_sec,
                              total_time_sec, n_genes; read by export_nonbar_plot_excel.py
  metrics_across_scaling.csv  one row per run: AUROC, AUPRC, MacroAUPRC, MacroF1, F1_case, F1_control,
                              conf_mat, seed, fold, patients, cells
  per_donor_predictions.csv   seed, fold, donor, true (class index), prob (of class index 1),
                              patients_used, cells_used
  confusion_matrices.npy      row-normalised 2x2 confusion matrices in run order
  line plots (mean and 95% CI over the 10 seed x fold runs per point): epoch_time_vs_cells.pdf,
    total_time_vs_cells.pdf, gpu_memory_vs_cells.pdf
  bar plots: total_cells_barplot.pdf, total_time_barplot.pdf, gpu_memory_barplot.pdf
  scatter plots with a least-squares line: gpu_memory_vs_total_cells.pdf, total_time_vs_total_cells.pdf
Checkpoints go to cellection's default ./disease_id_classification_SEED<seed>/, and each run overwrites
the previous run of the same seed.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import time

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

DECISION_THRESHOLD = 0.5

PLOT_PALETTE = [
    "#2C7FB8",  # blue
    "#D7301F",  # red
    "#41AB5D",  # green
    "#C51B8A",  # magenta
    "#1D91C0",  # cyan
    "#F16913",  # orange
]
sns.set_theme(style="white", palette=PLOT_PALETTE)


def subsample_patients_from_train(train_donors, n_patients):
    """Draw `n_patients` training donors, without replacement when there are enough of them.

    Otherwise the whole draw is with replacement, so some donors repeat and others can be left out.
    Uses the global np.random state.
    """
    replace = n_patients > len(train_donors)
    return np.random.choice(train_donors, size=n_patients, replace=replace)


def build_train_adata_with_dup(adata, sample_key, selected_donors):
    """Concatenate the cells of `selected_donors` in draw order, one bag per draw.

    Repeat draws are renamed "<id>_dup1", "<id>_dup2", ... in obs[sample_key], so each copy is a
    separate bag with the same cells and label as the original.
    """
    import anndata as ad
    parts = []
    donor_counts = {}
    for donor in selected_donors:
        donor_subset = adata[adata.obs[sample_key] == donor].copy()
        if donor in donor_counts:
            donor_counts[donor] += 1
            donor_subset.obs[sample_key] = f"{donor}_dup{donor_counts[donor]}"
        else:
            donor_counts[donor] = 0
        parts.append(donor_subset)
    return ad.concat(parts, axis=0, merge="same")


def add_best_fit_line(ax, x, y, color="black", linestyle="--"):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return
    coeffs = np.polyfit(x[mask], y[mask], 1)
    x_sorted = np.sort(x[mask])
    y_fit = coeffs[0] * x_sorted + coeffs[1]
    ax.plot(x_sorted, y_fit, color=color, linestyle=linestyle, linewidth=1.5)


def subsample_cells_per_patient(adata, target_cells, noise_scale=0.02):
    """Resize every donor bag in `adata` to exactly `target_cells` cells.

    Bags come from the hard-coded obs column "donor_id". Larger donors are subsampled without
    replacement. Smaller donors keep all their cells and are padded with noisy copies of randomly
    chosen cells of the same donor (N(0, noise_scale) per gene, clipped at 0). Padded cells carry only
    obs["donor_id"], no label; cellection reads a bag's label from its first cell, which is always real
    because real cells come first. concatenate() suffixes keep cell names unique. Uses the global
    np.random state.
    """
    import anndata as ad
    out = []

    for donor, donor_data in adata.obs.groupby("donor_id"):
        donor_subset = adata[adata.obs["donor_id"] == donor]

        if donor_subset.n_obs > target_cells:
            idx = np.random.choice(donor_subset.obs_names, target_cells, replace=False)
            out.append(donor_subset[idx])
            continue

        if donor_subset.n_obs == target_cells:
            out.append(donor_subset)
            continue

        # Too few cells: pad with noisy copies of real ones.
        n_existing = donor_subset.n_obs
        n_needed = target_cells - n_existing

        X = donor_subset.X.toarray() if hasattr(donor_subset.X, "toarray") else donor_subset.X
        real_cells = X

        chosen_idx = np.random.choice(np.arange(n_existing), n_needed, replace=True)

        augmented = []
        for i in chosen_idx:
            base = real_cells[i]
            noise = np.random.normal(0, noise_scale, base.shape)
            x_new = np.maximum(base + noise, 0)
            augmented.append(x_new)

        aug = ad.AnnData(np.vstack(augmented))
        aug.obs["donor_id"] = donor
        aug.var_names = donor_subset.var_names

        combined = donor_subset.concatenate(aug, batch_key=None)
        out.append(combined)

    return out[0].concatenate(*out[1:], batch_key=None)


def compute_metrics_binary(y_true, y_prob):
    """AUROC, AUPRC, MacroAUPRC, F1 scores and the raw confusion matrix, with class index 1 as positive.

    MacroAUPRC averages the PR-curve areas of class 1 (scored by y_prob) and class 0 (by 1 - y_prob).
    F1_case / F1_control use pos_label=1 / 0, and conf_mat rows are the true class.
    """
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    out = {}
    out["AUROC"] = roc_auc_score(y_true, y_prob)
    out["AUPRC"] = average_precision_score(y_true, y_prob)

    precision_pos, recall_pos, _ = precision_recall_curve(y_true, y_prob)
    precision_neg, recall_neg, _ = precision_recall_curve(1 - y_true, 1 - y_prob)

    out["MacroAUPRC"] = (auc(recall_pos, precision_pos) + auc(recall_neg, precision_neg)) / 2
    out["MacroF1"] = f1_score(y_true, y_pred, average="macro")
    out["F1_case"] = f1_score(y_true, y_pred, pos_label=1)
    out["F1_control"] = f1_score(y_true, y_pred, pos_label=0)
    out["conf_mat"] = confusion_matrix(y_true, y_pred)

    return out


adata = sc.read_h5ad(
    f"{DATA_DIR}/covid/covid_Ziegler_processed.h5ad"
)
n_genes = adata.shape[1]
print("Number of genes:", n_genes)
folds = pd.read_csv(
    f"{DATA_DIR}/covid/covid_5folds.csv",
    index_col=0
)

save_dir = f"{RESULTS_DIR}/R1-Q13/Code/Results/Covid"
os.makedirs(save_dir, exist_ok=True)

if "disease_id" not in adata.obs.columns:
    adata.obs["disease_id"] = "control"
    adata.obs.loc[
        adata.obs["disease__ontology_label"].isin(["COVID-19","long COVID-19"]),
        "disease_id"
    ] = "case"

sample_key = "donor_id"
task_key = "disease_id"
num_folds = folds["fold"].nunique()

donors = adata.obs[sample_key].unique()
N_total = len(donors)


# About 1/4, 1/2, 1x and 2x the 58 donors in the dataset (each fold trains on ~46-47 donors).
patients_grid = [14, 29, 58, 116]

print("Patients Grid:", patients_grid)

# Within one cell of the 25th, 50th and 75th percentiles and the maximum of cells per donor, i.e. the
# 8-bin edges 120 / 410 / 816 / 2140 from cell_count_imbalance/covid/before_downsampling.py.
cells_grid = [119, 409, 815, 2140]

print("Cells Grid:", cells_grid)


SEEDS = [0,1]   # 2 seeds x 5 folds = 10 runs per grid point
EPOCHS = 200    # epochs_run when cellection reports no early-stopping epoch

results = []
scaling_results = []
all_predictions = []

roc_curves = []
pr_curves = []
conf_mats = []


for SEED in SEEDS:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n==================== SEED {SEED} ====================")

    for fold in range(1, num_folds + 1):
        print(f"\n==================== FOLD {fold} ====================")

        fold_test = folds.loc[folds["fold"] == fold, "samples"].values
        fold_train = folds.loc[folds["fold"] != fold, "samples"].values

        for n_patients in patients_grid:

            # One patient draw per (seed, fold, n_patients), reused for all four bag sizes, so only
            # the bag size changes along cells_grid.
            chosen_train_patients = subsample_patients_from_train(fold_train, n_patients)

            for target_cells in cells_grid:

                print(f"\n► Scaling Run: {n_patients} patients × {target_cells} cells/patient")

                adata_train = build_train_adata_with_dup(
                    adata,
                    sample_key,
                    chosen_train_patients
                )
                adata_train = subsample_cells_per_patient(adata_train, target_cells)

                # All test donors of the fold, resized to the same bag size as training.
                adata_test = adata[adata.obs[sample_key].isin(fold_test)]
                adata_test = subsample_cells_per_patient(adata_test, target_cells)

                if adata_train.obs[sample_key].nunique() <= 2:
                    print("Skipping: insufficient train donors.")
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
                val_size=0.2,
                aggregator="gated_attention",
                global_features=256,
                attention_dim=64,
                max_epochs=200,
                learning_rate=1e-4,
                batch_size=15,
                seed=SEED,
                device=torch.device("cuda"),
                hidden_layer=[256,32],
                activation=nn.ReLU(),
                layernorm=True,
                dropout_rate=0.1,
                patience=20,
                save_model=True,
            )
                obj.prepare()

                # Only obj.train() is timed. Memory is torch's peak *allocated* CUDA memory since the
                # reset, which leaves out the CUDA context and cached-but-unused blocks.
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                obj.train()

                t1 = time.time()
                total_time = t1 - t0

                # cellection does not set obj.early_stop_epoch, so every run counts as EPOCHS (200)
                # epochs and epoch_time_sec is total_time_sec / 200, even when early stopping ended
                # training sooner.
                if hasattr(obj, "early_stop_epoch"):
                    epochs_run = obj.early_stop_epoch
                else:
                    epochs_run = EPOCHS

                epoch_time = total_time / epochs_run if epochs_run > 0 else np.nan

                # MiB, despite the _mb column name
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**2

                scaling_results.append({
                    "seed": SEED,
                    "fold": fold,
                    "train_patients": n_patients,
                    "cells_per_patient": target_cells,
                    "total_cells_train": adata_train.n_obs,
                    "gpu_memory_mb": gpu_mem,
                    "epochs_run": epochs_run,
                    "epoch_time_sec": epoch_time,
                    "total_time_sec": total_time,
                    "n_genes": n_genes,
                })

                obj.adata = adata_test
                obj.prepare_inference()

                _, true_df, pred_df, _, _, _ = obj.inference()

                # Columns are in alphabetical label order; column 1 is the positive class from here on.
                y_true = np.argmax(true_df.values, axis=1)
                y_prob = pred_df.values[:, 1]

                all_predictions.append(pd.DataFrame({
                    "seed": SEED,
                    "fold": fold,
                    "donor": pred_df.index,
                    "true": y_true,
                    "prob": y_prob,
                    "patients_used": n_patients,
                    "cells_used": target_cells
                }))

                m = compute_metrics_binary(y_true, y_prob)
                m["seed"] = SEED
                m["fold"] = fold
                m["patients"] = n_patients
                m["cells"] = target_cells
                results.append(m)

                # ROC and PR curves (kept in memory only; not saved or plotted)
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_curves.append({"fpr": fpr, "tpr": tpr})

                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                pr_curves.append({"precision": precision, "recall": recall})

                cm = m["conf_mat"].astype(float)
                cm = cm / cm.sum(axis=1, keepdims=True)
                conf_mats.append(cm)


pd.DataFrame(results).to_csv(f"{save_dir}/metrics_across_scaling.csv", index=False)
pd.DataFrame(scaling_results).to_csv(f"{save_dir}/scaling_runtime_gpu.csv", index=False)
pd.concat(all_predictions).to_csv(f"{save_dir}/per_donor_predictions.csv", index=False)

np.save(f"{save_dir}/confusion_matrices.npy", np.stack(conf_mats))

print("\nSaved all results!")


runtime_df = pd.DataFrame(scaling_results)
# Line plots: seaborn averages the 10 seed x fold rows per point and draws a 95% CI band. Scatter
# plots show every run with a least-squares line.

plt.figure(figsize=(7,5))
sns.lineplot(
    data=runtime_df,
    x="cells_per_patient",
    y="epoch_time_sec",
    hue="train_patients",
    palette=PLOT_PALETTE,
    marker="o"
)
plt.title("Average Epoch Time vs. Cells per Patient")
plt.xlabel("Cells per Patient")
plt.ylabel("Time per Epoch (sec)")
plt.legend(title="Num Patients")
plt.tight_layout()
plt.savefig(f"{save_dir}/epoch_time_vs_cells.pdf", dpi=200)
plt.close()

plt.figure(figsize=(7,5))
sns.lineplot(
    data=runtime_df,
    x="cells_per_patient",
    y="total_time_sec",
    hue="train_patients",
    palette=PLOT_PALETTE,
    marker="o"
)
plt.title("Total Training Time vs. Cells per Patient")
plt.xlabel("Cells per Patient")
plt.ylabel("Total Time (sec)")
plt.tight_layout()
plt.savefig(f"{save_dir}/total_time_vs_cells.pdf", dpi=200)
plt.close()

plt.figure(figsize=(7,5))
sns.lineplot(
    data=runtime_df,
    x="cells_per_patient",
    y="gpu_memory_mb",
    hue="train_patients",
    palette=PLOT_PALETTE,
    marker="o"
)
plt.title("GPU Memory Usage vs. Cells per Patient")
plt.xlabel("Cells per Patient")
plt.ylabel("Peak GPU Memory (MB)")
plt.tight_layout()
plt.savefig(f"{save_dir}/gpu_memory_vs_cells.pdf", dpi=200)
plt.close()

plt.figure(figsize=(7,5))
sns.barplot(
    data=runtime_df,
    x="train_patients",
    y="total_cells_train",
    hue="cells_per_patient",
    palette=PLOT_PALETTE
)
plt.title("Total Cells in Training per Scaling Condition")
plt.xlabel("Training Patients")
plt.ylabel("Total Cells in Training Set")
plt.tight_layout()
plt.savefig(f"{save_dir}/total_cells_barplot.pdf", dpi=200)
plt.close()

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=runtime_df,
    x="total_cells_train",
    y="gpu_memory_mb",
    hue="train_patients",
    size="cells_per_patient",
    sizes=(40, 200),
    palette=PLOT_PALETTE
)
plt.title("GPU Memory vs Total Cells in Training")
plt.xlabel("Total Cells (Train)")
plt.ylabel("GPU Peak Memory (MB)")
plt.legend(title="train_patients")
add_best_fit_line(plt.gca(), runtime_df["total_cells_train"].values, runtime_df["gpu_memory_mb"].values)
plt.tight_layout()
plt.savefig(f"{save_dir}/gpu_memory_vs_total_cells.pdf", dpi=200)
plt.close()

plt.figure(figsize=(7,5))
sns.barplot(
    data=runtime_df,
    x="train_patients",
    y="total_time_sec",
    hue="cells_per_patient",
    palette=PLOT_PALETTE
)
plt.title("Total Training Time per Scaling Condition")
plt.xlabel("Training Patients")
plt.ylabel("Total Training Time (sec)")
plt.tight_layout()
plt.savefig(f"{save_dir}/total_time_barplot.pdf", dpi=200)
plt.close()

plt.figure(figsize=(7,5))
sns.barplot(
    data=runtime_df,
    x="train_patients",
    y="gpu_memory_mb",
    hue="cells_per_patient",
    palette=PLOT_PALETTE
)
plt.title("GPU Memory Usage per Scaling Condition")
plt.xlabel("Training Patients")
plt.ylabel("GPU Peak Memory (MB)")
plt.tight_layout()
plt.savefig(f"{save_dir}/gpu_memory_barplot.pdf", dpi=200)
plt.close()

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=runtime_df,
    x="total_cells_train",
    y="total_time_sec",
    hue="train_patients",
    size="cells_per_patient",
    sizes=(40, 200),
    palette=PLOT_PALETTE
)
plt.title("Total Training Time vs Total Cells in Training")
plt.xlabel("Total Cells (Train)")
plt.ylabel("Total Training Time (sec)")
plt.legend(title="train_patients")
add_best_fit_line(plt.gca(), runtime_df["total_cells_train"].values, runtime_df["total_time_sec"].values)
plt.tight_layout()
plt.savefig(f"{save_dir}/total_time_vs_total_cells.pdf", dpi=200)
plt.close()

print("Saved ALL plots!")
