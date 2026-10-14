"""Simulated PBMC cohort: 10x PBMC 3k vs GSE132044 bags, with or without B cells.

The PBMC counterpart of simulate_pancreas.py, with two platforms as batches: "tenx" (10x PBMC 3k
raw counts) and "smartseq" (every cell of GSE132044 as written by ../build_gse132044_h5ad.py,
whatever its protocol). GSE132044 gene names ("<ENSEMBL>_<SYMBOL>") are cut to symbols and only
shared genes are kept. Cells are typed B / T / Mono / NK by Leiden clustering (resolution 0.5)
of both platforms pooled, each cluster taking the type whose marker (MS4A1, CD3D, CD14, NKG7)
has the highest mean.

Per platform: 20 bags of 500 cells drawn with replacement, S00-S09 no_b and S10-S19 with_b.
Cell-type counts per bag are a multinomial draw with equal probabilities over B / T / Mono / NK
(B excluded for no_b). RNG seed 0. Bags keep raw counts in .X; obs: orig_cell (source cell),
sample_id, batch, b_status, cell_type.

Reads $DATA_DIR/pbmc3k/pbmc3k_raw.h5ad and
$RESULTS_DIR/R1-q1/Data/pbmc_smartseq/pbmc_smartseq_raw.h5ad. Writes:
    $DATA_DIR/pbmc3k/pbmc_real_10x_vs_smartseq.h5ad, read by evaluate_pbmc.py
    $RESULTS_DIR/R1-q1/Results_pbmc_real_batch/
        sample_metadata.csv (bag, batch, label, n_cells), cells_per_batch.csv,
        cells_per_batch_by_celltype.csv/.png, bag_counts_per_batch.csv, bag_sizes.csv,
        bag_sizes_hist.png, cells_per_sample_by_celltype.csv
        umap_{batch,b_status,celltype}.png (before training)
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

PBMC10X_PATH = f"{DATA_DIR}/pbmc3k/pbmc3k_raw.h5ad"
SMARTSEQ_PATH = f"{RESULTS_DIR}/R1-q1/Data/pbmc_smartseq/pbmc_smartseq_raw.h5ad"
OUT_DATA_PATH = f"{DATA_DIR}/pbmc3k/pbmc_real_10x_vs_smartseq.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R1-q1/Results_pbmc_real_batch"
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

BATCH_NAMES = ["tenx", "smartseq"]
SAMPLES_PER_BATCH = 20
CELLS_PER_SAMPLE = 500
CELL_TYPES = ["B", "T", "Mono", "NK"]
RNG = np.random.default_rng(0)


def load_and_align():
    """Both PBMC platforms as one AnnData on their shared genes, raw counts in .X and layers["counts"].

    GSE132044 names are cut to the symbol; genes without one, and repeats after the first, are
    dropped. obs["batch"] is "tenx" or "smartseq".
    """
    pbmc = sc.read(PBMC10X_PATH)
    smart = sc.read(SMARTSEQ_PATH)

    smart.var["symbol"] = smart.var_names.to_series().str.split("_", n=1).str[1]
    smart = smart[:, smart.var["symbol"].notna()].copy()
    smart = smart[:, ~smart.var["symbol"].duplicated()].copy()
    smart.var_names = smart.var["symbol"]

    # pbmc3k var_names are already symbols
    common = pbmc.var_names.intersection(smart.var_names)
    pbmc = pbmc[:, common].copy()
    smart = smart[:, common].copy()

    pbmc.obs["batch"] = "tenx"
    smart.obs["batch"] = "smartseq"

    pbmc.layers["counts"] = pbmc.X.copy()
    smart.layers["counts"] = smart.X.copy()

    adata = ad.concat([pbmc, smart], join="inner", merge="first")
    return adata


def annotate_cell_types(adata):
    """obs["cell_type"] (B / T / Mono / NK) from Leiden clusters of both platforms pooled.

    Log-normalised, PCA, Leiden at resolution 0.5; each cluster takes the type whose marker gene has
    the highest mean log-expression. simulate_nk_hijack.py clusters each platform separately
    instead; its annotate_cell_types_per_batch() says why.
    """
    tmp = adata.copy()
    sc.pp.normalize_total(tmp)
    sc.pp.log1p(tmp)
    sc.pp.pca(tmp)
    sc.pp.neighbors(tmp)
    sc.tl.leiden(tmp, resolution=0.5)

    marker_genes = {
        "B": ["MS4A1"],
        "T": ["CD3D"],
        "Mono": ["CD14"],
        "NK": ["NKG7"],
    }
    cluster_means = (
        pd.DataFrame(
            tmp.X.toarray() if hasattr(tmp.X, "toarray") else tmp.X,
            columns=tmp.var_names,
            index=tmp.obs["leiden"],
        )
        .groupby(level=0)
        .mean()
    )
    cluster_to_type = {}
    for cluster in cluster_means.index:
        scores = {ct: cluster_means.loc[cluster, genes].mean() for ct, genes in marker_genes.items()}
        cluster_to_type[cluster] = max(scores, key=scores.get)
    adata.obs["cell_type"] = tmp.obs["leiden"].map(cluster_to_type)
    return adata


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


def build_sample(pool, sample_id, batch_name, remove_b):
    """One bag of CELLS_PER_SAMPLE cells drawn with replacement from `pool`, a single platform.

    Type counts are a multinomial draw over CELL_TYPES (no B if remove_b). .X holds raw counts.
    """
    # probs follow CELL_TYPES order, so index 0 is B
    probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    if remove_b:
        probs[0] = 0.0
        probs = probs / probs.sum()
    counts = RNG.multinomial(CELLS_PER_SAMPLE, probs)

    chosen = []
    for ct, n in zip(CELL_TYPES, counts):
        if n == 0:
            continue
        pool_ct = pool.obs_names[pool.obs["cell_type"] == ct]
        picked = RNG.choice(pool_ct, size=n, replace=True)
        chosen.extend(picked)

    ad_sub = pool[chosen].copy()
    ad_sub.obs["orig_cell"] = chosen
    ad_sub.obs["sample_id"] = sample_id
    ad_sub.obs["batch"] = batch_name
    ad_sub.obs["b_status"] = "no_b" if remove_b else "with_b"
    ad_sub.obs_names = [f"{sample_id}__{i}" for i in range(ad_sub.n_obs)]
    # CELLECTION takes raw counts
    if "counts" in ad_sub.layers:
        ad_sub.X = ad_sub.layers["counts"].copy()
    return ad_sub


def build_dataset(adata):
    """SAMPLES_PER_BATCH bags per platform, first half no_b, concatenated."""
    samples = []
    for batch_name in BATCH_NAMES:
        pool = adata[adata.obs["batch"] == batch_name].copy()
        for i in range(SAMPLES_PER_BATCH):
            remove_b = i < (SAMPLES_PER_BATCH // 2)
            sample_id = f"{batch_name}_S{i:02d}"
            samples.append(build_sample(pool, sample_id, batch_name, remove_b))
    return ad.concat(samples, join="outer", merge="first")


def summarize(sim_adata):
    """Bag metadata and cell- and bag-count tables, with plots, in SAVE_DIR."""
    meta = (
        sim_adata.obs.groupby(["sample_id", "batch", "b_status"])
        .size()
        .reset_index(name="n_cells")
    )
    meta.to_csv(os.path.join(SAVE_DIR, "sample_metadata.csv"), index=False)

    cells_per_batch = sim_adata.obs["batch"].value_counts().loc[BATCH_NAMES]
    cells_per_batch.to_csv(os.path.join(SAVE_DIR, "cells_per_batch.csv"))

    cells_per_batch_ct = (
        sim_adata.obs.groupby(["batch", "cell_type"])
        .size()
        .unstack(fill_value=0)
        .loc[BATCH_NAMES]
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
        .loc[BATCH_NAMES]
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
        sim_adata.obs.groupby(["sample_id", "cell_type"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    cells_per_sample_ct.to_csv(os.path.join(SAVE_DIR, "cells_per_sample_by_celltype.csv"))


def plot_umaps(sim_adata):
    """Pre-training UMAPs of the simulated cells, coloured by batch, b_status and cell type."""
    adata_umap = preprocess_for_umap(sim_adata)
    adata_umap.obs["batch"] = pd.Categorical(adata_umap.obs["batch"], categories=BATCH_NAMES)
    sc.pl.umap(
        adata_umap,
        color="batch",
        palette=["#1f77b4", "#ff7f0e"],
        title="PBMC 10x vs Smart-seq – colored by batch",
        show=False,
    )
    plt.savefig(os.path.join(SAVE_DIR, "umap_batch.png"), dpi=150, bbox_inches="tight")
    plt.close()

    sc.pl.umap(
        adata_umap,
        color="b_status",
        title="PBMC – with_b vs no_b",
        show=False,
    )
    plt.savefig(os.path.join(SAVE_DIR, "umap_b_status.png"), dpi=150, bbox_inches="tight")
    plt.close()

    sc.pl.umap(
        adata_umap,
        color="cell_type",
        title="PBMC – cell types",
        show=False,
    )
    plt.savefig(os.path.join(SAVE_DIR, "umap_celltype.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    adata = load_and_align()
    adata = annotate_cell_types(adata)
    print("Loaded and annotated real PBMC batches:", adata)
    sim_adata = build_dataset(adata)
    # under $DATA_DIR/pbmc3k/ rather than $RESULTS_DIR: evaluate_pbmc.py reads it
    sim_adata.write_h5ad(OUT_DATA_PATH)
    print("Simulated dataset:", sim_adata)
    summarize(sim_adata)
    plot_umaps(sim_adata)


if __name__ == "__main__":
    main()
