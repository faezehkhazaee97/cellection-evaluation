"""Does attention find the planted B cells? CELLECTION on the simulated PBMC3k cohort.

Input is $DATA_DIR/pbmc3k/pbmc3k_simulated_patients_Bcell_causal.h5ad from simulate_pbmc3k.py, with
bags in obs["patient_id"], labels in obs["disease"] (1 = case, 0 = control), obs["cell_type"] and a
UMAP in obsm. Cases and controls differ only in their B-cell fraction, so B cells
(obs["cell_type"] == "B") are the cells attention ought to pick out.

For each seed (6, 7, 4, 8, 3) a binary CELLECTION model (instance encoder, T-Net, gated attention)
is trained without an outer CV: the package holds out 20% of patients for early stopping, and
inference then runs on all patients, training ones included. Patient-level metrics are AUROC, AUPRC
and macro-F1 at a 0.5 threshold. At the cell level, the cells of each bag that has any B cells are
ranked by raw attention, giving cell AUPRC (B = positive), Enrich@k (fraction of B cells among the
top k%, k = 1, 5, 10) and BackgroundB (the bag's B fraction), each averaged over bags; fold
enrichment is Enrich@k / BackgroundB. Only control bags can lack B cells, so these averages cover
the cases plus every control that drew at least one B cell. UMAPs mark each patient's top-10%
attention cells (at or above that patient's TOP_ATTENTION_QUANTILE) over all cells and over
case-only and control-only subsets, where cases need a B fraction >= 0.30 and controls <= 0.05.

Writes to $RESULTS_DIR/R1-q4a/Results/:
  patient_level_metrics.csv         seed, AUROC, AUPRC, MacroF1
  cell_level_metrics.csv            seed, CellAUPRC, Enrich@{1,5,10}%, BackgroundB, Enrich@{1,5,10}%_FC
  dataset_case_control_sanity.csv   cell and patient counts of the case/control subsets and overlaps
  umap_panel_sanity_counts.csv      cells, patients and top-10% cells per seed and UMAP panel
  enrichment_barplot.pdf            Enrich@k (mean +/- SD over seeds) with the mean BackgroundB line
  umap_cell_types.pdf, seed_<S>_umap_top10_attention_{all,case,control}.pdf,
  seed_<S>_umap_top10_only_{case,control}.pdf
  seed_<S>/                         checkpoints
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import shutil

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve

import cellection

DATA_PATH = (
    f"{DATA_DIR}/pbmc3k/"
    "pbmc3k_simulated_patients_Bcell_causal.h5ad"
)

SAVE_DIR = (
    f"{RESULTS_DIR}/R1-q4a/Results"
)
os.makedirs(SAVE_DIR, exist_ok=True)

SEEDS = [6, 7, 4, 8, 3]

SAMPLE_KEY = "patient_id"
TASK_KEY = "disease"          # binary: 0 control, 1 case
CAUSAL_CELLTYPE = "B"         # the planted subpopulation
CASE_B_FRAC_MIN = 0.30        # simulated case bags have 30-40% B cells
CONTROL_B_FRAC_MAX = 0.05     # simulated control bags have 0-5% B cells
DECISION_THRESHOLD = 0.5      # case probability cut-off for macro-F1
TOP_ATTENTION_QUANTILE = 0.90  # per-patient attention quantile defining the "Top 10%" cells

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_h5ad_safe(path, scratch_dir):
    """sc.read_h5ad with a workaround for uns entries stored with encoding_type='null'.

    Some anndata versions cannot read uns["log1p"] / uns["pca"] written that way. The file is then
    copied to scratch_dir as *_nolog1p.h5ad (once), the two entries are deleted from the copy, and
    the copy is read. Other read errors are re-raised.
    """
    try:
        return sc.read_h5ad(path)
    except Exception as exc:
        msg = str(exc)
        if "encoding_type='null'" not in msg:
            raise
        fixed_path = os.path.join(
            scratch_dir, os.path.basename(path).replace(".h5ad", "_nolog1p.h5ad")
        )
        if not os.path.exists(fixed_path):
            shutil.copyfile(path, fixed_path)
        with h5py.File(fixed_path, "a") as f:
            if "uns" in f:
                if "log1p" in f["uns"]:
                    del f["uns"]["log1p"]
                if "pca" in f["uns"]:
                    del f["uns"]["pca"]
        return sc.read_h5ad(fixed_path)


adata = read_h5ad_safe(DATA_PATH, SAVE_DIR)
adata.obs_names_make_unique()

print("\n=== DATA SUMMARY ===")
print("Cells:", adata.n_obs)
print("Genes:", adata.n_vars)
print("Patients:", adata.obs[SAMPLE_KEY].nunique())
print("Disease counts:\n", adata.obs[TASK_KEY].value_counts())
print("Cell types:\n", adata.obs["cell_type"].value_counts())
adata.obs["case_control"] = adata.obs[TASK_KEY].map({0: "control", 1: "case"})
patient_b_frac = adata.obs.groupby(SAMPLE_KEY)["cell_type"].apply(
    lambda x: (x == CAUSAL_CELLTYPE).mean()
)
adata.obs["b_frac_patient"] = adata.obs[SAMPLE_KEY].map(patient_b_frac)
print("Patients with b_frac >= 0.30:", (patient_b_frac >= CASE_B_FRAC_MIN).sum())
print("Patients with b_frac <= 0.05:", (patient_b_frac <= CONTROL_B_FRAC_MAX).sum())
# A cell is "case" (or "control") only if both its label and its patient's realised B fraction fit
# the simulated range. These sets define the case-only / control-only UMAP panels and the sanity
# table below.
case_patients = patient_b_frac[patient_b_frac >= CASE_B_FRAC_MIN].index
control_patients = patient_b_frac[patient_b_frac <= CONTROL_B_FRAC_MAX].index
case_mask_strict = (
    (adata.obs["case_control"] == "case")
    & (adata.obs[SAMPLE_KEY].isin(case_patients))
)
control_mask_strict = (
    (adata.obs["case_control"] == "control")
    & (adata.obs[SAMPLE_KEY].isin(control_patients))
)
case_cells = adata.obs.index[case_mask_strict]
control_cells = adata.obs.index[control_mask_strict]
print("Case cells:", len(case_cells))
print("Control cells:", len(control_cells))
print("Overlap cells:", len(case_cells.intersection(control_cells)))
print("Overlap patients:", len(set(case_patients) & set(control_patients)))

dataset_sanity_df = pd.DataFrame(
    [
        {
            "dataset_cells_total": adata.n_obs,
            "dataset_patients_total": adata.obs[SAMPLE_KEY].nunique(),
            "case_cells_strict": int(case_mask_strict.sum()),
            "control_cells_strict": int(control_mask_strict.sum()),
            "case_patients_strict": adata.obs.loc[case_mask_strict, SAMPLE_KEY].nunique(),
            "control_patients_strict": adata.obs.loc[control_mask_strict, SAMPLE_KEY].nunique(),
            "case_control_cell_overlap": len(set(case_cells) & set(control_cells)),
            "case_control_patient_overlap": len(set(case_patients) & set(control_patients)),
        }
    ]
)
dataset_sanity_df.to_csv(f"{SAVE_DIR}/dataset_case_control_sanity.csv", index=False)
print("Saved:", f"{SAVE_DIR}/dataset_case_control_sanity.csv")

plt.figure(figsize=(6, 5))

sc.pl.umap(
    adata,
    color="cell_type",
    palette="tab10",
    size=10,
    alpha=0.9,
    show=False
)

plt.title("Cell types")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/umap_cell_types.pdf")
plt.close()


def enrichment_at_k(scores, labels, k_frac):
    """Fraction of positives among the top k_frac of cells by score (at least one cell)."""
    k = max(1, int(len(scores) * k_frac))
    idx = np.argsort(scores)[-k:]
    return labels[idx].mean()


patient_metrics = []
cell_metrics = []

all_pr_curves = []
all_roc_curves = []
panel_sanity_rows = []

for SEED in SEEDS:
    print(f"\n==============================")
    print(f"Running SEED {SEED}")
    print(f"==============================")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    obj = cellection.cellectiion_object(
        adata=adata.copy(),
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
        val_size=0.2,
        max_epochs=200,
        patience=20,

        seed=SEED,
        device=DEVICE,

        save_model=True,
        save_path=f"{SAVE_DIR}/seed_{SEED}"
    )

    obj.prepare()
    obj.train()

    # Inference runs on every patient, including those used for training.
    obj.prepare_inference()
    (
        sample_meta,
        true_df,
        pred_df,
        embedding_df,
        global_feature_df,
        attn_df
    ) = obj.inference()
    # attn_df: one row per cell of every patient (index = cell name); "attention_score_raw" is the
    # attention weight after the softmax over that patient's cells (sums to 1 within a bag).
    # true_df / pred_df: one row per patient; true_df is one-hot and pred_df holds class
    # probabilities, column 1 = case.
    attn_df["patient_id"] = adata.obs.loc[attn_df.index, SAMPLE_KEY].values

    y_true = np.argmax(true_df.values, axis=1)
    y_prob = pred_df.values[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    patient_metrics.append({
        "seed": SEED,
        "AUROC": auroc,
        "AUPRC": auprc,
        "MacroF1": macro_f1
    })

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    all_roc_curves.append((fpr, tpr))

    # Cell level: within each bag, does attention rank the B cells first?
    attn_df["cell_type"] = adata.obs.loc[attn_df.index, "cell_type"]
    attn_df["is_causal"] = (attn_df["cell_type"] == CAUSAL_CELLTYPE).astype(int)

    per_patient_metrics = []

    for pid, df_p in attn_df.groupby(SAMPLE_KEY):
        scores = df_p["attention_score_raw"].values
        labels = df_p["is_causal"].values

        if labels.sum() == 0:
            continue  # only control bags can lack B cells

        per_patient_metrics.append({
            "CellAUPRC": average_precision_score(labels, scores),
            "Enrich@1%": enrichment_at_k(scores, labels, 0.01),
            "Enrich@5%": enrichment_at_k(scores, labels, 0.05),
            "Enrich@10%": enrichment_at_k(scores, labels, 0.10),
            "BackgroundB": labels.mean()
        })

    df_pp = pd.DataFrame(per_patient_metrics)
    cell_metrics.append({
        "seed": SEED,
        "CellAUPRC": df_pp["CellAUPRC"].mean(),
        "Enrich@1%": df_pp["Enrich@1%"].mean(),
        "Enrich@5%": df_pp["Enrich@5%"].mean(),
        "Enrich@10%": df_pp["Enrich@10%"].mean(),
        "BackgroundB": df_pp["BackgroundB"].mean()
    })

    # UMAPs: each patient's top-10% attention cells against a case/control background.
    adata_all = adata.copy()
    adata_all.obs["attention_group"] = (
        adata_all.obs["case_control"]
        .map({"case": "Case background", "control": "Control background"})
        .astype(str)
    )
    for pid, df_p in attn_df.groupby(SAMPLE_KEY):
        idx = df_p.index.intersection(adata_all.obs.index)
        if idx.empty:
            continue
        scores = df_p.loc[idx, "attention_score_raw"]
        thresh = np.quantile(scores, TOP_ATTENTION_QUANTILE)
        adata_all.obs.loc[idx[scores >= thresh], "attention_group"] = "Top 10%"
    n_top_all = int((adata_all.obs["attention_group"] == "Top 10%").sum())
    n_case_all = int((adata_all.obs["case_control"] == "case").sum())
    n_control_all = int((adata_all.obs["case_control"] == "control").sum())
    panel_sanity_rows.append(
        {
            "seed": SEED,
            "panel": "all",
            "n_cells": adata_all.n_obs,
            "n_patients": adata_all.obs[SAMPLE_KEY].nunique(),
            "n_top10": n_top_all,
            "n_background": int(adata_all.n_obs - n_top_all),
        }
    )
    print(
        f"[SANITY][seed={SEED}][all] n_cells={adata_all.n_obs}, "
        f"n_patients={adata_all.obs[SAMPLE_KEY].nunique()}, n_top10={n_top_all}"
    )

    sc.pl.umap(
        adata_all,
        color="attention_group",
        palette={
            "Case background": "#b0b0b0",
            "Control background": "#d9d9d9",
            "Top 10%": "red",
        },
        size=10,
        alpha=0.8,
        show=False,
        legend_loc=None,
    )
    plt.title(
        f"Top scoring cells (all)\nN={adata_all.n_obs}, case={n_case_all}, "
        f"control={n_control_all}, Top10={n_top_all}"
    )
    plt.tight_layout()
    plt.savefig(
        f"{SAVE_DIR}/seed_{SEED}_umap_top10_attention_all.pdf",
        bbox_inches="tight",
    )
    plt.close()

    for label, tag in [("case", "case"), ("control", "control")]:
        if label == "case":
            adata_subset = adata[
                (adata.obs["case_control"] == label)
                & (adata.obs[SAMPLE_KEY].isin(case_patients))
            ].copy()
        else:
            adata_subset = adata[
                (adata.obs["case_control"] == label)
                & (adata.obs[SAMPLE_KEY].isin(control_patients))
            ].copy()
        if adata_subset.n_obs == 0:
            print(f"[WARN] No cells for {tag}.")
            continue

        adata_subset.obs["attention_group"] = "Background"
        adata_subset.obs["attention_group"] = adata_subset.obs["attention_group"].astype(str)
        subset_patients = set(adata_subset.obs[SAMPLE_KEY].unique())
        for pid, df_p in attn_df.groupby(SAMPLE_KEY):
            if pid not in subset_patients:
                continue
            idx = df_p.index.intersection(adata_subset.obs.index)
            if idx.empty:
                continue
            scores = df_p.loc[idx, "attention_score_raw"]
            thresh = np.quantile(scores, TOP_ATTENTION_QUANTILE)
            adata_subset.obs.loc[idx[scores >= thresh], "attention_group"] = "Top 10%"
        n_top_subset = int((adata_subset.obs["attention_group"] == "Top 10%").sum())
        panel_sanity_rows.append(
            {
                "seed": SEED,
                "panel": tag,
                "n_cells": adata_subset.n_obs,
                "n_patients": adata_subset.obs[SAMPLE_KEY].nunique(),
                "n_top10": n_top_subset,
                "n_background": int(adata_subset.n_obs - n_top_subset),
            }
        )
        print(
            f"[SANITY][seed={SEED}][{tag}] n_cells={adata_subset.n_obs}, "
            f"n_patients={adata_subset.obs[SAMPLE_KEY].nunique()}, n_top10={n_top_subset}"
        )

        # The grey background is this subset's cells only.
        bg_color = "#9e9e9e" if label == "case" else "#dddddd"
        bg_tag = "case-only background" if label == "case" else "control-only background"
        sc.pl.umap(
            adata_subset,
            color="attention_group",
            palette={"Background": bg_color, "Top 10%": "red"},
            size=12,
            alpha=0.8,
            show=False,
            legend_loc=None,
        )
        plt.title(
            f"Top scoring cells ({tag})\nN={adata_subset.n_obs}, Top10={n_top_subset}, {bg_tag}"
        )
        plt.tight_layout()
        plt.savefig(
            f"{SAVE_DIR}/seed_{SEED}_umap_top10_attention_{tag}.pdf",
            bbox_inches="tight",
        )
        plt.close()

        # Same panel with only the top-10% cells (no background)
        ad_top = adata_subset[adata_subset.obs["attention_group"] == "Top 10%"].copy()
        if ad_top.n_obs > 0:
            sc.pl.umap(
                ad_top,
                color="attention_group",
                palette={"Top 10%": "red"},
                size=18,
                alpha=0.9,
                show=False,
                legend_loc=None,
            )
            plt.title(f"Top scoring cells ({tag} only)\nN={ad_top.n_obs}")
            plt.tight_layout()
            plt.savefig(
                f"{SAVE_DIR}/seed_{SEED}_umap_top10_only_{tag}.pdf",
                bbox_inches="tight",
            )
            plt.close()

df_patient = pd.DataFrame(patient_metrics)
df_cell = pd.DataFrame(cell_metrics)
df_cell["Enrich@1%_FC"] = df_cell["Enrich@1%"] / df_cell["BackgroundB"]
df_cell["Enrich@5%_FC"] = df_cell["Enrich@5%"] / df_cell["BackgroundB"]
df_cell["Enrich@10%_FC"] = df_cell["Enrich@10%"] / df_cell["BackgroundB"]
df_patient.to_csv(f"{SAVE_DIR}/patient_level_metrics.csv", index=False)
df_cell.to_csv(f"{SAVE_DIR}/cell_level_metrics.csv", index=False)
df_panel_sanity = pd.DataFrame(panel_sanity_rows)
df_panel_sanity.to_csv(f"{SAVE_DIR}/umap_panel_sanity_counts.csv", index=False)
print("Saved:", f"{SAVE_DIR}/umap_panel_sanity_counts.csv")

print("\n=== PATIENT-LEVEL (MEAN ± SD) ===")
print(df_patient.mean())
print(df_patient.std())

print("\n=== CELL-LEVEL (MEAN ± SD) ===")
print(df_cell.mean())
print(df_cell.std())


enrich_df = df_cell.melt(
    id_vars="seed",
    value_vars=["Enrich@1%", "Enrich@5%", "Enrich@10%"],
    var_name="k",
    value_name="fraction_B"
)
enrich_df["k"] = enrich_df["k"].str.replace("Enrich@", "", regex=False)

fig, ax = plt.subplots(figsize=(5,4))
sns.barplot(data=enrich_df, x="k", y="fraction_B", ci="sd", ax=ax)
ax.axhline(
    df_cell["BackgroundB"].mean(),
    linestyle="--",
    color="red",
    label="Background B-cell freq (cases)"
)
ax.set_ylabel("Fraction of B cells")
ax.set_title("Enrichment of causal B cells", pad=12)
ax.set_xlabel("Top fraction")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=True)
fig.tight_layout(rect=[0, 0.15, 1, 1])
fig.savefig(f"{SAVE_DIR}/enrichment_barplot.pdf", bbox_inches="tight")
plt.close(fig)

print("\n✅ DONE. All reviewer-required results generated.")
