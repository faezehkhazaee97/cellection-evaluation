"""Simulated pancreas cohort: Baron vs Muraro bags, with or without beta cells.

Each bag (synthetic patient) is drawn from one study only, so study is a pure technical batch,
and its label b_status (with_b / no_b) is whether it contains beta cells. Both labels occur in
both studies, so batch says nothing about the label. Per study: 20 bags of 500 cells sampled
with replacement from its alpha, beta, delta and gamma cells; S00-S09 are no_b (166-167 cells of
each other type), S10-S19 with_b (125 of each type). RNG seed 0. Adds obs sample_id, batch and
b_status; obs_names are "<sample_id>__<i>".

Reads $DATA_DIR/pancreatic_w_batch/Fig2_obj.h5ad (obs["dataset"] = study, obs["celltype"]).
Writes:
    $DATA_DIR/pancreatic_w_batch/pancreas_synthetic_baron_muraro.h5ad, read by
        evaluate_pancreas.py and evaluate_pancreas_single_seed.py
    $RESULTS_DIR/R1-q1/Results_pancreas_baron_muraro_synth/
        cells_per_batch.csv/.png, cells_per_batch_by_celltype.csv/.png,
        bag_counts_per_batch.csv (bags per batch x label), bag_sizes.csv, bag_sizes_hist.png,
        cells_per_sample_by_celltype.csv
        umap_{batch,b_status,celltype}.png: the batch effect before training (log-normalised,
        2,000 HVGs, scaled, PCA)
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

DATA_PATH = f"{DATA_DIR}/pancreatic_w_batch/Fig2_obj.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R1-q1/Results_pancreas_baron_muraro_synth"
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

BATCHES = ["baron", "muraro"]
CELL_TYPE_KEY = "celltype"
B_CELL_LABEL = "beta"
CELL_TYPES = ["alpha", "beta", "delta", "gamma"]
SAMPLES_PER_BATCH = 20
CELLS_PER_SAMPLE = 500
RNG = np.random.default_rng(0)


def preprocess_for_umap(adata):
    """Copy of `adata` with a UMAP: log-normalise, 2,000 HVGs, scale, PCA, 15-NN graph on 30 PCs."""
    adata = adata.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata, init_pos="random", random_state=0)
    return adata


def fixed_counts_per_type(target_celltypes, total_cells):
    """Split total_cells evenly over the types; the first `remainder` types get one extra cell."""
    n_types = len(target_celltypes)
    base = total_cells // n_types
    counts = np.array([base] * n_types, dtype=int)
    remainder = total_cells - base * n_types
    if remainder > 0:
        counts[:remainder] += 1
    return counts


def build_sample(pool, sample_id, batch_name, remove_b, target_celltypes):
    """One bag of CELLS_PER_SAMPLE cells drawn with replacement from `pool`, a single study.

    No beta cells if remove_b. Sets obs sample_id, batch, b_status and obs_names "<sample_id>__<i>".
    """
    n_types = len(target_celltypes)
    # 500 / 4 = 125 of each type. For no_b, beta goes to 0 and the other types are rescaled to
    # 500 (floor, then +1 to the largest counts until the total is 500).
    counts = fixed_counts_per_type(target_celltypes, CELLS_PER_SAMPLE)
    if B_CELL_LABEL in target_celltypes:
        b_idx = target_celltypes.index(B_CELL_LABEL)
        if remove_b:
            counts[b_idx] = 0
            non_b = counts.sum()
            if non_b == 0:
                raise ValueError("All counts are zero after removing beta cells.")
            counts = (counts / non_b) * CELLS_PER_SAMPLE
            counts = np.floor(counts).astype(int)
            diff = CELLS_PER_SAMPLE - counts.sum()
            if diff > 0:
                order = np.argsort(-counts)
                counts[order[:diff]] += 1

    chosen = []
    for ct, n in zip(target_celltypes, counts):
        if n == 0:
            continue
        pool_ct = pool.obs_names[pool.obs[CELL_TYPE_KEY] == ct]
        picked = RNG.choice(pool_ct, size=n, replace=True)
        chosen.extend(picked)

    ad_sub = pool[chosen].copy()
    ad_sub.obs["sample_id"] = sample_id
    ad_sub.obs["batch"] = batch_name
    ad_sub.obs["b_status"] = "no_b" if remove_b else "with_b"
    ad_sub.obs_names = [f"{sample_id}__{i}" for i in range(ad_sub.n_obs)]
    return ad_sub


def build_synthetic(adata):
    """SAMPLES_PER_BATCH bags per study in BATCHES, first half no_b, concatenated."""
    samples = []
    for batch_name in BATCHES:
        pool = adata[adata.obs["dataset"] == batch_name].copy()
        target_celltypes = [ct for ct in CELL_TYPES if ct in pool.obs[CELL_TYPE_KEY].unique()]
        for i in range(SAMPLES_PER_BATCH):
            remove_b = i < (SAMPLES_PER_BATCH // 2)
            sample_id = f"{batch_name}_S{i:02d}"
            samples.append(build_sample(pool, sample_id, batch_name, remove_b, target_celltypes))
    return sc.concat(samples, join="outer", merge="first")


def summarize(sim_adata):
    """Cell- and bag-count tables of the simulated cohort, with bar plots, in SAVE_DIR."""
    sim_adata.obs["batch"] = pd.Categorical(sim_adata.obs["batch"], categories=BATCHES)

    cells_per_batch = sim_adata.obs["batch"].value_counts().loc[BATCHES]
    cells_per_batch.to_csv(os.path.join(SAVE_DIR, "cells_per_batch.csv"))
    ax = cells_per_batch.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Cells per batch")
    ax.set_ylabel("Cells")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "cells_per_batch.png"), dpi=150)
    plt.close()

    cells_per_batch_ct = (
        sim_adata.obs.groupby(["batch", CELL_TYPE_KEY])
        .size()
        .unstack(fill_value=0)
        .loc[BATCHES]
    )
    cells_per_batch_ct.to_csv(os.path.join(SAVE_DIR, "cells_per_batch_by_celltype.csv"))
    ax = cells_per_batch_ct.T.plot(kind="bar", stacked=True)
    ax.set_title("Cell-type composition per batch")
    ax.set_ylabel("Cells")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "cells_per_batch_by_celltype.png"), dpi=150)
    plt.close()

    bag_counts = (
        sim_adata.obs.groupby(["batch", "b_status"])["sample_id"]
        .nunique()
        .unstack(fill_value=0)
        .loc[BATCHES]
    )
    bag_counts.to_csv(os.path.join(SAVE_DIR, "bag_counts_per_batch.csv"))

    bag_sizes = sim_adata.obs.groupby("sample_id").size()
    bag_sizes.to_csv(os.path.join(SAVE_DIR, "bag_sizes.csv"))
    plt.hist(bag_sizes, bins=20, color="gray")
    plt.xlabel("Cells per bag")
    plt.ylabel("Bags")
    plt.title("Bag size distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "bag_sizes_hist.png"), dpi=150)
    plt.close()

    cells_per_sample_ct = (
        sim_adata.obs.groupby(["sample_id", CELL_TYPE_KEY], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    cells_per_sample_ct.to_csv(os.path.join(SAVE_DIR, "cells_per_sample_by_celltype.csv"))


def plot_umaps(sim_adata):
    """Pre-training UMAPs of the simulated cells, coloured by batch, b_status and cell type."""
    adata_umap = preprocess_for_umap(sim_adata)
    sc.pl.umap(
        adata_umap,
        color="batch",
        palette=["#1f77b4", "#ff7f0e"],
        title="Pancreas synthetic – colored by batch",
        show=False,
    )
    plt.savefig(os.path.join(SAVE_DIR, "umap_batch.png"), dpi=150, bbox_inches="tight")
    plt.close()

    sc.pl.umap(
        adata_umap,
        color="b_status",
        title="Pancreas synthetic – with_b vs no_b",
        show=False,
    )
    plt.savefig(os.path.join(SAVE_DIR, "umap_b_status.png"), dpi=150, bbox_inches="tight")
    plt.close()

    sc.pl.umap(
        adata_umap,
        color=CELL_TYPE_KEY,
        title="Pancreas synthetic – cell types",
        show=False,
    )
    plt.savefig(os.path.join(SAVE_DIR, "umap_celltype.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    adata = sc.read_h5ad(DATA_PATH)
    adata = adata[adata.obs["dataset"].isin(BATCHES)].copy()
    sim_adata = build_synthetic(adata)
    # under $DATA_DIR rather than $RESULTS_DIR: the evaluate scripts read it as input
    sim_adata.write_h5ad(f"{DATA_DIR}/pancreatic_w_batch/pancreas_synthetic_baron_muraro.h5ad")
    summarize(sim_adata)
    plot_umaps(sim_adata)
    print(sim_adata)


if __name__ == "__main__":
    main()
