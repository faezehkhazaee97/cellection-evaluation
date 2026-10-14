"""Does batch-blind CELLECTION attention follow the batch-confounded NK cells?

Input: the bags from simulate_nk_hijack.py, where NK cells are 5% of non-B cells in tenx and 30%
in smartseq but the label (b_status) depends only on B cells. The model gets no batch
information (batch_key=None). For seeds 3, 4, 6, 7, 8 it is trained on all bags (val_size=0.2
held back for early stopping; instance encoder [256, 32], T-Net on, gated attention, up to 200
epochs, patience 20) and then run on all bags, training ones included, to get each cell's
attention.

Per bag: NK and B fractions, the NK share among the top 1 / 5 / 10% attention cells, the
cell-level AUPRC of attention for NK, and for with_b bags the same AUPRC and top-1% share for B.
Per batch: the mean NK metrics, and the top-1% / top-5% NK share divided by the batch's mean NK
fraction. Attention drawn to the NK excess shows up as a higher fold change in smartseq. The
B-cell summary over with_b bags (printed only) checks that attention still finds the cells that
set the label.

Reads $DATA_DIR/pbmc3k/pbmc_batch_confounded_nk_hijack.h5ad. Writes to
$RESULTS_DIR/R1-q1/Results_attention_hijack_test/:
    attn_df_seed_<s>.csv (per-cell attention with bag, cell type, batch, label)
    per_patient_hijack_metrics.csv (one row per seed x bag)
    nk_hijack_summary_by_batch.csv, nk_hijack_foldchange_by_batch.csv
    models_seed_<s>/
Submitted with submit_nk_hijack_test.sbatch; uses a GPU when available.
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
from sklearn.metrics import average_precision_score

import cellection

DATA_PATH = f"{DATA_DIR}/pbmc3k/pbmc_batch_confounded_nk_hijack.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R1-q1/Results_attention_hijack_test"
os.makedirs(SAVE_DIR, exist_ok=True)

SAMPLE_KEY = "sample_id"
TASK_KEY = "b_status"
BATCH_KEY = "batch"
CELLTYPE_KEY = "cell_type"
CAUSAL_CELLTYPE = "B"
DECOY_CELLTYPE = "NK"
DECOY_BATCH = "smartseq"  # batch where NK is over-represented
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [3, 4, 6, 7, 8]


def enrichment_at_k(scores, labels, k_frac):
    """Share of positive labels among the top k_frac of cells by score (at least one cell)."""
    k = max(1, int(len(scores) * k_frac))
    idx = np.argsort(scores)[-k:]
    return labels[idx].mean()


def main():
    adata = sc.read_h5ad(DATA_PATH)
    adata.obs_names_make_unique()
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype("category")
    adata.obs[TASK_KEY] = adata.obs[TASK_KEY].astype("category")
    print("cells:", adata.n_obs, "patients:", adata.obs[SAMPLE_KEY].nunique())
    print(adata.obs.groupby([BATCH_KEY, TASK_KEY]).size())

    all_rows = []
    patient_metrics_rows = []

    for SEED in SEEDS:
        print(f"\n===== seed {SEED} =====")
        obj = cellection.cellectiion_object(
            adata=adata.copy(),
            task_type="classification",
            task_key=TASK_KEY,
            sample_key=SAMPLE_KEY,
            batch_key=None,  # batch-blind by design: model never sees batch
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
            save_path=f"{SAVE_DIR}/models_seed_{SEED}",
        )
        obj.prepare()
        obj.train()

        # obj.adata is still the full dataset, so attention is scored on every bag, training bags
        # included.
        obj.prepare_inference()
        sample_meta, true_df, pred_df, embedding_df, global_feature_df, attn_df = obj.inference()

        attn_df = attn_df.copy()
        attn_df["sample_id"] = adata.obs.loc[attn_df.index, SAMPLE_KEY].values
        attn_df["cell_type"] = adata.obs.loc[attn_df.index, CELLTYPE_KEY].values
        attn_df["batch"] = adata.obs.loc[attn_df.index, BATCH_KEY].values
        attn_df["b_status"] = adata.obs.loc[attn_df.index, TASK_KEY].values
        attn_df["is_B"] = (attn_df["cell_type"] == CAUSAL_CELLTYPE).astype(int)
        attn_df["is_NK"] = (attn_df["cell_type"] == DECOY_CELLTYPE).astype(int)

        attn_df.to_csv(f"{SAVE_DIR}/attn_df_seed_{SEED}.csv")

        for pid, df_p in attn_df.groupby(SAMPLE_KEY, observed=True):
            scores = df_p["attention_score_raw"].values
            batch = df_p["batch"].iloc[0]
            b_status = df_p["b_status"].iloc[0]
            is_b = df_p["is_B"].values
            is_nk = df_p["is_NK"].values
            row = {
                "seed": SEED,
                "patient_id": pid,
                "batch": batch,
                "b_status": b_status,
                "n_cells": len(scores),
                "nk_frac": is_nk.mean(),
                "b_frac": is_b.mean(),
                "NK_Enrich@1%": enrichment_at_k(scores, is_nk, 0.01),
                "NK_Enrich@5%": enrichment_at_k(scores, is_nk, 0.05),
                "NK_Enrich@10%": enrichment_at_k(scores, is_nk, 0.10),
            }
            if is_nk.sum() > 0 and is_nk.sum() < len(is_nk):
                row["NK_CellAUPRC"] = average_precision_score(is_nk, scores)
            else:
                row["NK_CellAUPRC"] = np.nan
            if is_b.sum() > 0 and is_b.sum() < len(is_b):
                row["B_CellAUPRC"] = average_precision_score(is_b, scores)
                row["B_Enrich@1%"] = enrichment_at_k(scores, is_b, 0.01)
            else:
                row["B_CellAUPRC"] = np.nan
                row["B_Enrich@1%"] = np.nan
            patient_metrics_rows.append(row)

    df = pd.DataFrame(patient_metrics_rows)
    df.to_csv(f"{SAVE_DIR}/per_patient_hijack_metrics.csv", index=False)

    print("\n\n=== NK (decoy, phenotype-irrelevant) enrichment among top-attention cells, by batch ===")
    summary = df.groupby(["batch"]).agg(
        n_patients=("patient_id", "nunique"),
        mean_nk_frac=("nk_frac", "mean"),
        NK_Enrich_1pct=("NK_Enrich@1%", "mean"),
        NK_Enrich_5pct=("NK_Enrich@5%", "mean"),
        NK_Enrich_10pct=("NK_Enrich@10%", "mean"),
        NK_CellAUPRC=("NK_CellAUPRC", "mean"),
    )
    print(summary)
    summary.to_csv(f"{SAVE_DIR}/nk_hijack_summary_by_batch.csv")

    print("\n=== Fold-change: NK enrichment relative to that batch's own NK background rate ===")
    fc = summary.copy()
    fc["NK_Enrich_1pct_FC"] = fc["NK_Enrich_1pct"] / fc["mean_nk_frac"]
    fc["NK_Enrich_5pct_FC"] = fc["NK_Enrich_5pct"] / fc["mean_nk_frac"]
    print(fc)
    fc.to_csv(f"{SAVE_DIR}/nk_hijack_foldchange_by_batch.csv")

    print("\n=== Sanity check: B-cell (real causal signal) recovery, case patients only ===")
    b_case = df[df.b_status == "with_b"]
    print(b_case.groupby("batch").agg(B_CellAUPRC=("B_CellAUPRC", "mean"), B_Enrich_1pct=("B_Enrich@1%", "mean")))


if __name__ == "__main__":
    main()
