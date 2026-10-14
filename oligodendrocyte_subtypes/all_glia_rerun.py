"""Same pipeline as oligodendrocyte_rerun.py, with all four SEA-AD MTG glial classes in the model.

Astrocytes, microglia-PVM, oligodendrocytes and OPCs stay together here (the cells of the all-glia
embedding file, $SEAAD_GLIA_DIR/MTG/MTG_sciMultiLaMA_CELL_embedding.csv), so oligodendrocytes
compete with the other lineages for attention. The model, its settings and the per-seed analysis
are unchanged: the competition table covers all four classes, while the top-10% UMAPs and the
supertype enrichment look at oligodendrocytes only. Inference covers every donor, so the printed
AUROC and macro F1 include the training donors.

Otherwise only three things change: SEEDS are 6-9 and 15-20, outputs go to
$RESULTS_DIR/R1-q12/Results/test/, and figures are saved as .png instead of .pdf, under the same
file names as in oligodendrocyte_rerun.py. As there, y_true == 1 is the second obs["disease"]
category in pd.get_dummies order and y_pred = y_prob > 0.5.

Reads $DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad (obs donor_id, disease, cell_type,
Supertype) and the embedding above. Needs the `cellection` package; uses a GPU when available.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root
SEAAD_GLIA_DIR = os.environ.get("SEAAD_GLIA_DIR", os.path.join(DATA_DIR, "SEAAD_Glia"))  # embedding CSVs


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from sklearn.metrics import f1_score, roc_auc_score

import cellection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = f"{DATA_DIR}/sea-ad/MTG_glia_intersection_norm.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R1-q12/Results/test"
os.makedirs(SAVE_DIR, exist_ok=True)

adata = sc.read_h5ad(DATA_PATH)
adata.obs_names_make_unique()

sci_path = f"{SEAAD_GLIA_DIR}/MTG/MTG_sciMultiLaMA_CELL_embedding.csv"
sci = pd.read_csv(sci_path, index_col=0)
# Only cells present in the embedding file are kept, for training as well as plotting.
common_ids = adata.obs_names.intersection(sci.index)
if common_ids.empty:
    raise ValueError("No overlapping cell IDs between adata and sciMultiLaMA embeddings.")
adata = adata[common_ids].copy()
sci = sci.loc[common_ids]
adata.obsm["X_sciMultiLaMA"] = sci.values
print("Loaded sciMultiLaMA embeddings:", adata.obsm["X_sciMultiLaMA"].shape)
sc.pp.neighbors(adata, n_neighbors=25, use_rep="X_sciMultiLaMA")
sc.tl.umap(adata, min_dist=0.3, random_state=0)
celltype_palette = {
    "Astrocyte": "#E64B35",
    "Microglia–PVM": "#4DBBD5",
    "Oligodendrocyte": "#3C5488",
    "OPC": "#F39B7F"
}
SEEDS = [6,7,8,9,15,16,17,18,19,20]
TARGET_CELLTYPE = "oligodendrocyte"
TOP_FRAC = 0.10
TOP_N = 100
USE_PRED_AD = False
# Per-donor attention quantile: cells at or above it are that donor's top-10% attention cells.
TOP_ATTENTION_QUANTILE = 0.90

for SEED in SEEDS:
    print(f"\n=== SEED {SEED} ===")

    obj = cellection.cellectiion_object(
        adata=adata.copy(),

        task_type="classification",
        task_key="disease",           # AD vs Control
        sample_key="donor_id",
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
        batch_size=3,
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
    # Re-point the object at the shared adata so inference covers every donor (training and
    # validation). inference() writes attention columns into adata.obs, so they are dropped first
    # to avoid clashes with the previous seed's columns.
    obj.adata = adata
    for col in ["attention_score_raw", "attention_score_norm_cellnum"]:
        if col in obj.adata.obs.columns:
            obj.adata.obs = obj.adata.obs.drop(columns=[col])
    obj.prepare_inference()
    (
        sample_meta,
        true_df,
        pred_df,
        embedding_df,
        global_feature_df,
        attn_df
    ) = obj.inference()
    meta = adata.obs.reindex(attn_df.index)
    attn_df["donor_id"] = meta["donor_id"].values
    attn_df["cell_type"] = meta["cell_type"].values
    attn_df["Supertype"] = meta["Supertype"].values
    attn_df = attn_df.dropna(subset=["donor_id"])

    # All donors' oligodendrocytes, on the UMAP computed above.
    adata_oligo = adata[adata.obs["cell_type"] == TARGET_CELLTYPE].copy()

    sample_meta["donor_id"] = sample_meta.index
    sample_meta_all = sample_meta.copy()

    # true_df / pred_df: one row per donor, disease categories as columns in pd.get_dummies order;
    # y_true == 1 and y_prob refer to the second column.
    y_true = np.argmax(true_df.values, axis=1)
    y_prob = pred_df.values[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    pred_summary = pd.DataFrame({
        "donor_id": true_df.index,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred
    }).set_index("donor_id")
    correct_ad_donors = set(
        pred_summary.index[
            (pred_summary["y_true"] == 1)
            & (pred_summary["y_pred"] == 1)
        ]
    )
    true_ad_donors = set(pred_summary.index[pred_summary["y_true"] == 1])
    true_non_ad_donors = set(pred_summary.index[pred_summary["y_true"] == 0])

    print("AUROC:", roc_auc_score(y_true, y_prob))
    print("MacroF1:", f1_score(y_true, y_pred, average="macro"))
    # Glial cell-type shares among each correctly predicted y_true == 1 donor's top-10% attention
    # cells (top) and among all of its cells (bg).
    results = []
    for donor, df_d in attn_df.groupby("donor_id", observed=True):
        if donor not in correct_ad_donors:
            continue

        scores = df_d["attention_score_raw"].values
        thresh = np.quantile(scores, TOP_ATTENTION_QUANTILE)

        top = df_d[scores >= thresh]
        bg = df_d

        for ct in ["microglial cell","oligodendrocyte","astrocyte of the cerebral cortex","oligodendrocyte precursor cell"]:
            results.append({
                "seed": SEED,
                "donor": donor,
                "cell_type": ct,
                "top_frac": (top["cell_type"] == ct).mean(),
                "bg_frac": (bg["cell_type"] == ct).mean()
            })

    df_comp = pd.DataFrame(results)
    df_comp.to_csv(f"{SAVE_DIR}/competition_seed_{SEED}.csv", index=False)

    def plot_top10_umap(donor_ids, tag, base_adata, restrict_celltype=None):
        """Mark each donor's top-10% attention cells on the UMAP of `donor_ids`' cells.

        With `restrict_celltype`, the top 10% is taken among cells of that type only.
        """
        attn_subset = attn_df[attn_df["donor_id"].isin(donor_ids)].copy()
        if restrict_celltype is not None:
            attn_subset = attn_subset[
                attn_subset["cell_type"] == restrict_celltype
            ]
        if attn_subset.empty:
            print(f"[WARN] No donors for {tag}.")
            return
        adata_subset = base_adata[
            base_adata.obs["donor_id"].isin(donor_ids)
        ].copy()
        adata_subset.obs["attention_group"] = "Background"
        top_cells = []
        for donor, df_d in attn_subset.groupby("donor_id", observed=True):
            scores = df_d["attention_score_raw"].values
            if scores.size == 0:
                continue
            thresh = np.quantile(scores, TOP_ATTENTION_QUANTILE)
            top_cells.extend(df_d.index[scores >= thresh].tolist())
        top_cells = adata_subset.obs.index.intersection(top_cells)
        adata_subset.obs.loc[top_cells, "attention_group"] = "Top 10%"
        plt.figure(figsize=(6,5))
        sc.pl.umap(
            adata_subset,
            color="attention_group",
            palette={
                "Background": "#D3D3D3",
                "Top 10%": "#E64B35"
            },
            size=10,
            alpha=0.85,
            show=False
        )
        plt.title("Top scoring cells")
        plt.tight_layout()
        plt.savefig(
            f"{SAVE_DIR}/umap_embedding_top10_seed_{SEED}_{tag}.png",
            dpi=300
        )
        plt.close()

    plot_top10_umap(
        true_ad_donors,
        "true_ad_all",
        adata_oligo,
        restrict_celltype=TARGET_CELLTYPE
    )
    plot_top10_umap(
        true_non_ad_donors,
        "true_non_ad_all",
        adata_oligo,
        restrict_celltype=TARGET_CELLTYPE
    )

    # Supertype enrichment among the top-TOP_N oligodendrocytes of the y_true == 1 donors.
    ad_donors = true_ad_donors

    attn_ad = attn_df[attn_df["donor_id"].isin(ad_donors)].copy()
    attn_oligo = attn_ad[attn_ad["cell_type"] == TARGET_CELLTYPE].copy()

    if attn_oligo.empty:
        print(f"[WARN] No {TARGET_CELLTYPE} cells found for AD donors.")
        continue

    top_oligo_parts = []
    for donor, df_d in attn_oligo.groupby("donor_id", observed=True):
        if df_d.empty:
            continue
        top_oligo_parts.append(df_d.nlargest(TOP_N, "attention_score_raw"))
    if not top_oligo_parts:
        print(f"[WARN] No top {TOP_N} {TARGET_CELLTYPE} cells found.")
        continue
    top_oligo = pd.concat(top_oligo_parts, axis=0)

    # Supertype shares among all oligodendrocytes of these donors vs. among the pooled per-donor
    # top-TOP_N cells; enrich = top share / overall share.
    super_all = attn_oligo["Supertype"].value_counts(normalize=True)
    super_top = top_oligo["Supertype"].value_counts(normalize=True)
    enrich = (super_top / super_all).fillna(0)
    target_super = enrich.idxmax()
    attn_target_super = attn_oligo[attn_oligo["Supertype"] == target_super]
    top_oligo_super = attn_target_super.nlargest(TOP_N, "attention_score_raw")

    adata_oligo.obs["attention_group"] = "Background"
    adata_oligo.obs.loc[top_oligo_super.index, "attention_group"] = f"Top {TOP_N}"

    plt.figure(figsize=(6, 5))
    sc.pl.umap(
        adata_oligo,
        color="Supertype",
        palette="tab10",
        size=10,
        alpha=0.9,
        show=False
    )
    plt.title("Oligodendrocyte embedding")
    plt.tight_layout()
    plt.savefig(
        f"{SAVE_DIR}/umap_oligo_subtypes_seed_{SEED}.png",
        dpi=300
    )
    plt.close()

    # UMAP marking the TOP_N highest-attention cells of the most enriched supertype
    plt.figure(figsize=(6, 5))
    sc.pl.umap(
        adata_oligo,
        color="attention_group",
        palette={
            "Background": "#D3D3D3",
            f"Top {TOP_N}": "#E64B35"
        },
        size=10,
        alpha=0.85,
        show=False
    )
    plt.title(f"{target_super} top {TOP_N} cells")
    plt.tight_layout()
    plt.savefig(
        f"{SAVE_DIR}/umap_oligo_top{TOP_N}_seed_{SEED}.png",
        dpi=300
    )
    plt.close()

    # Bar plot of every oligodendrocyte supertype: share among all vs. among top-TOP_N cells.
    all_supers = sorted(set(super_all.index).union(super_top.index))
    all_supers = [s for s in all_supers if "oligo" in str(s).lower()]
    all_pct = [super_all.get(s, 0) * 100 for s in all_supers]
    top_pct = [super_top.get(s, 0) * 100 for s in all_supers]

    print(
        f"[INFO] Most enriched oligo supertype: {target_super} "
        f"(ratio={enrich[target_super]:.2f})"
    )

    x = np.arange(len(all_supers))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, all_pct, width, color="lightgrey", label="All oligos")
    ax.bar(x + width / 2, top_pct, width, color="crimson", label=f"Top {TOP_N}")
    ax.set_xticks(x)
    ax.set_xticklabels(all_supers, rotation=30, ha="right")
    ax.set_ylabel("Fraction (%)")
    ax.set_title("Oligodendrocyte enrichment by supertype")
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(
        f"{SAVE_DIR}/oligo_enrichment_all_subtypes_seed_{SEED}.png",
        dpi=300
    )

    for super_name, all_val, top_val in zip(all_supers, all_pct, top_pct):
        fig, ax = plt.subplots(figsize=(4, 4))
        bars = ax.bar(
            [0, 1],
            [all_val, top_val],
            color=["lightgrey", "crimson"]
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"all {TARGET_CELLTYPE}", f"top {TOP_N} scored cells"]
        )
        ax.set_ylabel("Fraction (%)")
        ax.set_title(f"{super_name} enrichment")
        max_val = max(all_val, top_val)
        y_max = max_val * 1.15 if max_val > 0 else 1
        ax.set_ylim(0, y_max)
        label_offset = max_val * 0.03 if max_val > 0 else 0.1
        ax.axvline(0.5, linestyle="--", color="grey", linewidth=1)
        for bar in bars:
            height = bar.get_height()
            label_y = min(height + label_offset, y_max * 0.98)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{height:.1f}%",
                ha="center",
                va="bottom"
            )
        safe_name = super_name.replace(" ", "_").replace("/", "_")
        plt.tight_layout()
        fig.savefig(
            f"{SAVE_DIR}/oligo_enrichment_{safe_name}_seed_{SEED}.png",
            dpi=300
        )
