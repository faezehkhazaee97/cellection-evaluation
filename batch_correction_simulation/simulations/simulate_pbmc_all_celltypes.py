"""simulate_pbmc.py with fixed, equal cell-type counts per bag instead of multinomial draws.

Same platforms, gene alignment, pooled-clustering cell types and RNG seed (0). Differences: each
bag holds equal counts of every cell type found in its platform (in no_b bags B is set to 0 and
the rest scaled back up to 500), there are 10 bags per platform (S00-S04 no_b, S05-S09 with_b),
and plots are saved as PDF. simulate_nk_hijack.py imports load_and_align() from here.

Reads $DATA_DIR/pbmc3k/pbmc3k_raw.h5ad and
$RESULTS_DIR/R1-q1/Data/pbmc_smartseq/pbmc_smartseq_raw.h5ad. Writes:
    $DATA_DIR/pbmc3k/pbmc_real_10x_vs_smartseq_all_celltypes.h5ad, read by
        evaluate_pbmc_all_celltypes.py
    $RESULTS_DIR/R1-q1/Results_pbmc_real_batch_all_celltypes/
        sample_metadata.csv, cells_per_batch.csv, cells_per_batch_by_celltype.csv/.pdf,
        bag_counts_per_batch.csv, bag_sizes.csv, bag_sizes_hist.pdf,
        cells_per_sample_by_celltype.csv, umap_{batch,b_status,celltype}.pdf
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
OUT_DATA_PATH = f"{DATA_DIR}/pbmc3k/pbmc_real_10x_vs_smartseq_all_celltypes.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R1-q1/Results_pbmc_real_batch_all_celltypes"
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

BATCH_NAMES = ["tenx", "smartseq"]
SAMPLES_PER_BATCH = 10
CELLS_PER_SAMPLE = 500
B_CELL_LABEL = "B"
RNG = np.random.default_rng(0)


def load_and_align():
    """Both PBMC platforms as one AnnData on their shared genes, raw counts in .X and layers["counts"].

    GSE132044 names are cut to the symbol; genes without one, and repeats after the first, are
    dropped. obs["batch"] is "tenx" or "smartseq". Also imported by simulate_nk_hijack.py.
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
    Returns `adata` unchanged if obs already has cell_type.
    """
    if "cell_type" in adata.obs:
        return adata
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
    """One bag of CELLS_PER_SAMPLE cells drawn with replacement from `pool`, a single platform.

    Equal counts of each type in target_celltypes (no B if remove_b). .X holds raw counts.
    """
    # Equal counts of each type in this platform; for no_b, B goes to 0 and the rest are
    # rescaled to 500.
    counts = fixed_counts_per_type(target_celltypes, CELLS_PER_SAMPLE)
    if B_CELL_LABEL in target_celltypes:
        b_idx = target_celltypes.index(B_CELL_LABEL)
        if remove_b:
            counts[b_idx] = 0
            non_b = counts.sum()
            if non_b == 0:
                raise ValueError("All counts are zero after removing B cells.")
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
        target_celltypes = sorted(pool.obs["cell_type"].unique().tolist())
        for i in range(SAMPLES_PER_BATCH):
            remove_b = i < (SAMPLES_PER_BATCH // 2)
            sample_id = f"{batch_name}_S{i:02d}"
            samples.append(build_sample(pool, sample_id, batch_name, remove_b, target_celltypes))
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
    plt.savefig(os.path.join(SAVE_DIR, "cells_per_batch_by_celltype.pdf"), dpi=150)
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
    plt.savefig(os.path.join(SAVE_DIR, "bag_sizes_hist.pdf"), dpi=150)
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
    plt.savefig(os.path.join(SAVE_DIR, "umap_batch.pdf"), dpi=150, bbox_inches="tight")
    plt.close()

    sc.pl.umap(
        adata_umap,
        color="b_status",
        title="PBMC – with_b vs no_b",
        show=False,
    )
    plt.savefig(os.path.join(SAVE_DIR, "umap_b_status.pdf"), dpi=150, bbox_inches="tight")
    plt.close()

    sc.pl.umap(
        adata_umap,
        color="cell_type",
        title="PBMC – cell types",
        show=False,
    )
    plt.savefig(os.path.join(SAVE_DIR, "umap_celltype.pdf"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    adata = load_and_align()
    adata = annotate_cell_types(adata)
    print("Loaded and annotated real PBMC batches:", adata)
    sim_adata = build_dataset(adata)
    # under $DATA_DIR/pbmc3k/ rather than $RESULTS_DIR: evaluate_pbmc_all_celltypes.py reads it
    sim_adata.write_h5ad(OUT_DATA_PATH)
    print("Simulated dataset:", sim_adata)
    summarize(sim_adata)
    plot_umaps(sim_adata)


if __name__ == "__main__":
    main()
