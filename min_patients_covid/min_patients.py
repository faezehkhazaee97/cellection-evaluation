"""How few labelled patients can CELLECTION learn from? Patient subsampling on Haniffa COVID-19.

Severity is collapsed to a binary label in obs["severity_binary"]: Healthy and Non-covid are 0,
Severe, Critical and Death are 1, and cells with any other Worst_Clinical_Status are dropped. Bags
are obs["patient_id"], and a patient's label is the mode over its cells.

For each seed (0-4) and each total in TRAIN_PATIENTS_GRID (5-80) we draw a near-balanced set of
patients without replacement (for odd totals a coin flip gives the extra patient to cases or
controls) and train binary CELLECTION on it (instance encoder, T-Net, gated attention) with
val_size=0.3. Metrics are computed on the patients cellection put in its validation split. Those
same patients drive early stopping; there is no separate test set. Totals that need more cases or
controls than exist are skipped. AUROC and AUPRC are NaN when the validation patients are all one
class; macro-F1 uses a 0.5 threshold.

Reads $DATA_DIR/haniffa21_intersection.h5ad. Writes to
$RESULTS_DIR/R1-q10/Results/Haniffa_min_patients/:
  haniffa_binary_min_patients_metrics.csv  one row per seed and total: train and validation patient
                                           counts (total, case, control), AUROC, AUPRC, MacroF1
  haniffa_binary_min_patients_plot.pdf     each metric averaged over seeds vs number of patients
  seed_<S>_n<N>/                           checkpoints for each run
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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import cellection

DATA_PATH = f"{DATA_DIR}/haniffa21_intersection.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R1-q10/Results/Haniffa_min_patients"
os.makedirs(SAVE_DIR, exist_ok=True)

SAMPLE_KEY = "patient_id"
SEVERITY_KEY = "Worst_Clinical_Status"

# The two classes; cells with any other status are dropped.
NEGATIVE_LEVELS = ["Healthy", "Non-covid"]
POSITIVE_LEVELS = ["Severe", "Critical", "Death"]

TASK_KEY = "severity_binary"

TRAIN_PATIENTS_GRID = [5, 10, 15, 20, 25, 30, 35, 40, 60, 80]  # total patients (near-balanced)
SEEDS = [0, 1, 2, 3, 4]
VAL_FRACTION = 0.3
DECISION_THRESHOLD = 0.5  # case probability cut-off for macro-F1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_binary_labels(adata):
    """Add the 0/1 label in obs[TASK_KEY] and return a copy without the unlabelled cells."""
    adata.obs["severity_clean"] = (
        adata.obs[SEVERITY_KEY].astype(str).str.strip()
    )
    adata.obs[TASK_KEY] = np.nan
    adata.obs.loc[adata.obs["severity_clean"].isin(NEGATIVE_LEVELS), TASK_KEY] = 0
    adata.obs.loc[adata.obs["severity_clean"].isin(POSITIVE_LEVELS), TASK_KEY] = 1
    adata = adata[~adata.obs[TASK_KEY].isna()].copy()
    adata.obs[TASK_KEY] = adata.obs[TASK_KEY].astype(int)
    return adata


def get_patient_labels(adata):
    """One label per patient (mode over its cells); warns if a patient's cells disagree."""
    label_nunique = adata.obs.groupby(SAMPLE_KEY)[TASK_KEY].nunique()
    mixed = label_nunique[label_nunique > 1]
    if len(mixed) > 0:
        print("[WARN] Mixed labels for some patients; using mode.")
    patient_labels = (
        adata.obs.groupby(SAMPLE_KEY)[TASK_KEY]
        .agg(lambda x: x.mode().iloc[0])
    )
    return patient_labels


def compute_metrics_binary(y_true, y_prob):
    """AUROC and AUPRC (NaN if y_true has one class) and macro-F1 at DECISION_THRESHOLD."""
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    if len(np.unique(y_true)) < 2:
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "MacroF1": f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0),
    }


def get_val_patient_ids(obj, adata_source):
    """Patients in cellection's validation split.

    obj.val_dataset holds bag indices, so each bag of obj.mil_dataset is mapped to a patient through
    its first cell, looked up in adata_source.obs.
    """
    bag_patient_ids = []
    for metadata_indices in obj.mil_dataset.metadata_indices:
        first_cell = metadata_indices[0]
        bag_patient_ids.append(adata_source.obs.loc[first_cell, SAMPLE_KEY])
    bag_patient_ids = np.array(bag_patient_ids)
    val_bag_indices = np.array(obj.val_dataset.indices)
    val_patient_ids = np.unique(bag_patient_ids[val_bag_indices])
    return val_patient_ids


adata = sc.read_h5ad(DATA_PATH)
adata = build_binary_labels(adata)

print("Binary label counts:")
print(adata.obs[TASK_KEY].value_counts())

patient_labels = get_patient_labels(adata)
patient_ids = patient_labels.index.values

case_patients = patient_labels[patient_labels == 1].index.values
control_patients = patient_labels[patient_labels == 0].index.values

print("Patients total:", len(patient_ids))
print("Patients case:", len(case_patients))
print("Patients control:", len(control_patients))

results = []

for SEED in SEEDS:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    train_patients = patient_ids

    train_case = [p for p in train_patients if patient_labels.loc[p] == 1]
    train_ctrl = [p for p in train_patients if patient_labels.loc[p] == 0]

    for n_total in TRAIN_PATIENTS_GRID:
        # For odd totals a coin flip decides which class gets the extra patient.
        n_case = n_total // 2
        n_ctrl = n_total // 2
        if n_total % 2 == 1:
            if rng.random() < 0.5:
                n_case += 1
            else:
                n_ctrl += 1

        if n_case > len(train_case) or n_ctrl > len(train_ctrl):
            print(f"[SKIP] n_total={n_total} exceeds available train patients.")
            continue

        chosen_case = rng.choice(train_case, size=n_case, replace=False)
        chosen_ctrl = rng.choice(train_ctrl, size=n_ctrl, replace=False)
        chosen_train = np.concatenate([chosen_case, chosen_ctrl])

        adata_train = adata[adata.obs[SAMPLE_KEY].isin(chosen_train)].copy()

        print("\n==============================")
        print(f"SEED {SEED} | Train patients: {n_total} (case {n_case}, control {n_ctrl})")
        print("Train cells:", adata_train.n_obs)
        print("==============================")

        obj = cellection.cellectiion_object(
            adata=adata_train,
            task_type="classification",
            task_key=TASK_KEY,
            sample_key=SAMPLE_KEY,
            batch_key=None,

            input_type="measurement",
            sparse_input=True,

            InstanceEncoder=True,
            use_tnet=True,

            aggregator="gated_attention",
            global_features=256,
            attention_dim=64,
            hidden_layer=[256, 32],

            learning_rate=1e-4,
            batch_size=15,
            val_size=VAL_FRACTION,
            max_epochs=200,
            patience=20,

            seed=SEED,
            device=DEVICE,

            save_model=True,
            save_path=os.path.join(SAVE_DIR, f"seed_{SEED}_n{n_total}")
        )

        obj.prepare()
        val_patient_ids = get_val_patient_ids(obj, adata_train)
        val_case = int((patient_labels.loc[val_patient_ids] == 1).sum())
        val_ctrl = int((patient_labels.loc[val_patient_ids] == 0).sum())
        obj.train()

        # Score only the validation patients, with the best checkpoint.
        adata_val = adata_train[adata_train.obs[SAMPLE_KEY].isin(val_patient_ids)].copy()
        obj.adata = adata_val
        obj.prepare_inference()
        _, val_true_df, val_pred_df, _, _, _ = obj.inference()

        val_y_true = np.argmax(val_true_df.values, axis=1)
        val_y_prob = val_pred_df.values[:, 1]
        metrics = compute_metrics_binary(val_y_true, val_y_prob)

        results.append({
            "seed": SEED,
            "train_patients_total": n_total,
            "train_patients_case": n_case,
            "train_patients_control": n_ctrl,
            "val_patients_total": len(val_patient_ids),
            "val_patients_case": val_case,
            "val_patients_control": val_ctrl,
            **metrics,
        })

df = pd.DataFrame(results)
df.to_csv(os.path.join(SAVE_DIR, "haniffa_binary_min_patients_metrics.csv"), index=False)
print("\nSaved metrics to:", os.path.join(SAVE_DIR, "haniffa_binary_min_patients_metrics.csv"))

mean_df = (
    df.groupby("train_patients_total")[["AUROC", "AUPRC", "MacroF1"]]
    .mean()
    .reset_index()
)
std_df = (
    df.groupby("train_patients_total")[["AUROC", "AUPRC", "MacroF1"]]
    .std()
    .reset_index()
)

plt.figure(figsize=(6, 4))
for metric, color in zip(
    ["AUROC", "AUPRC", "MacroF1"],
    ["#1f77b4", "#ff7f0e", "#2ca02c"]
):
    plt.plot(
        mean_df["train_patients_total"],
        mean_df[metric],
        marker="o",
        linewidth=1.5,
        label=metric,
        color=color,
    )

plt.xlabel("Number of training patients (total)")
plt.ylabel("Performance")
plt.title("validation performance vs  number of patients")
plt.ylim(0, 1)
plt.legend(frameon=False)
plt.tight_layout()
plot_path = os.path.join(SAVE_DIR, "haniffa_binary_min_patients_plot.pdf")
plt.savefig(plot_path)
plt.close()
print("Saved plot to:", plot_path)
