"""Can a batch-driven cell population pull attention away from the cells that set the label?

PBMC bags in which NK abundance depends only on the platform and the label only on B cells.
NK cells make up NK_FRAC_BY_BATCH of the non-B cells (5% in tenx, 30% in smartseq); with_b bags
are B_FRAC_WITH_B = 20% B cells and no_b bags have none; the remaining cells are split equally
over the other types found in that platform. Per platform: 10 bags of 500 cells drawn with
replacement, S00-S04 no_b and S05-S09 with_b. RNG seed 0; CPU only. run_nk_hijack_test.py
trains and scores the model on the output.

The two platforms come from load_and_align() in simulate_pbmc_all_celltypes.py, which has to sit
next to this file ("tenx" = 10x PBMC 3k, "smartseq" = all GSE132044 cells). Cell types come from
Leiden clustering within each platform plus one marker gene per type (MARKER_GENES).

Reads $DATA_DIR/pbmc3k/pbmc3k_raw.h5ad and
$RESULTS_DIR/R1-q1/Data/pbmc_smartseq/pbmc_smartseq_raw.h5ad (from ../build_gse132044_h5ad.py).
Writes the bags (raw counts in .X; obs orig_cell, sample_id, batch, b_status, cell_type) to
$DATA_DIR/pbmc3k/pbmc_batch_confounded_nk_hijack.h5ad, and to
$RESULTS_DIR/R1-q1/Results_attention_hijack_test/: sample_composition.csv (cells per type in
each bag), celltype_counts_by_batch.csv, celltype_frac_by_batch.csv.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

# Make the sibling simulate_pbmc_all_celltypes.py importable from any working directory.
sys.path.insert(0, os.path.dirname(__file__))
from simulate_pbmc_all_celltypes import load_and_align

MARKER_GENES = {
    "B": ["MS4A1"],
    "T": ["CD3D"],
    "Mono": ["CD14"],
    "NK": ["NKG7"],
}


def annotate_cell_types_per_batch(adata):
    """obs["cell_type"] (B / T / Mono / NK) from Leiden clusters (resolution 0.5) within each batch.

    Each cluster takes the type whose MARKER_GENES have the highest mean log-expression. Pooled
    clustering of the 10x and GSE132044 cells groups them largely by platform (it labelled <1% of
    tenx cells but >60% of smartseq cells as NK), so each batch is clustered on its own. The NK
    confound in this simulation is added on purpose afterwards, in build_sample_confounded().
    """
    annotated = []
    for batch_name, sub in adata.obs.groupby("batch"):
        tmp = adata[sub.index].copy()
        sc.pp.normalize_total(tmp)
        sc.pp.log1p(tmp)
        sc.pp.pca(tmp)
        sc.pp.neighbors(tmp)
        sc.tl.leiden(tmp, resolution=0.5)

        cluster_means = (
            pd.DataFrame(
                tmp.X.toarray() if hasattr(tmp.X, "toarray") else tmp.X,
                columns=tmp.var_names,
                index=tmp.obs["leiden"],
            )
            .groupby(level=0, observed=True)
            .mean()
        )
        cluster_to_type = {}
        for cluster in cluster_means.index:
            scores = {ct: cluster_means.loc[cluster, genes].mean() for ct, genes in MARKER_GENES.items()}
            cluster_to_type[cluster] = max(scores, key=scores.get)
        cell_type = tmp.obs["leiden"].map(cluster_to_type)
        print(f"[{batch_name}] cluster sizes -> cell type:")
        print(cell_type.value_counts())
        annotated.append(cell_type)
    full_cell_type = pd.concat(annotated).loc[adata.obs.index]
    adata.obs["cell_type"] = full_cell_type
    return adata


OUT_DATA_PATH = f"{DATA_DIR}/pbmc3k/pbmc_batch_confounded_nk_hijack.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R1-q1/Results_attention_hijack_test"
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

BATCH_NAMES = ["tenx", "smartseq"]
SAMPLES_PER_BATCH = 10
CELLS_PER_SAMPLE = 500
B_CELL_LABEL = "B"
NK_LABEL = "NK"
B_FRAC_WITH_B = 0.20
NK_FRAC_BY_BATCH = {"tenx": 0.05, "smartseq": 0.30}
RNG = np.random.default_rng(0)


def build_sample_confounded(pool, sample_id, batch_name, remove_b, target_celltypes):
    """One bag of CELLS_PER_SAMPLE cells from `pool` (one batch), drawn with replacement.

    Returns (bag AnnData with raw counts in .X, {cell type: number of cells drawn}).
    """
    # B = B_FRAC_WITH_B or 0; NK = the batch's NK fraction x (1 - B), i.e. a fixed share of the
    # non-B cells; the rest is split equally over the other types in this batch.
    other_types = [c for c in target_celltypes if c not in (B_CELL_LABEL, NK_LABEL)]
    nk_frac = NK_FRAC_BY_BATCH[batch_name]
    b_frac = 0.0 if remove_b else B_FRAC_WITH_B
    remaining = 1.0 - b_frac
    nk_frac_scaled = nk_frac * remaining
    other_frac_each = (remaining - nk_frac_scaled) / len(other_types)

    fracs = {B_CELL_LABEL: b_frac, NK_LABEL: nk_frac_scaled}
    for ct in other_types:
        fracs[ct] = other_frac_each

    counts = {ct: int(round(f * CELLS_PER_SAMPLE)) for ct, f in fracs.items()}
    diff = CELLS_PER_SAMPLE - sum(counts.values())
    if diff != 0:
        # rounding error goes to the largest other type, so B and NK counts stay on target
        adj_key = max(other_types, key=lambda c: counts[c])
        counts[adj_key] += diff

    chosen = []
    chosen_types = []
    for ct, n in counts.items():
        if n <= 0:
            continue
        pool_ct = pool.obs_names[pool.obs["cell_type"] == ct]
        picked = RNG.choice(pool_ct, size=n, replace=True)
        chosen.extend(picked)
        chosen_types.extend([ct] * n)

    ad_sub = pool[chosen].copy()
    ad_sub.obs["orig_cell"] = chosen
    ad_sub.obs["sample_id"] = sample_id
    ad_sub.obs["batch"] = batch_name
    ad_sub.obs["b_status"] = "no_b" if remove_b else "with_b"
    ad_sub.obs_names = [f"{sample_id}__{i}" for i in range(ad_sub.n_obs)]
    # CELLECTION takes raw counts
    if "counts" in ad_sub.layers:
        ad_sub.X = ad_sub.layers["counts"].copy()
    return ad_sub, counts


def build_dataset_confounded(adata):
    """Concatenated bags for both batches (first half no_b); writes sample_composition.csv."""
    samples = []
    summary_rows = []
    for batch_name in BATCH_NAMES:
        pool = adata[adata.obs["batch"] == batch_name].copy()
        target_celltypes = sorted(pool.obs["cell_type"].unique().tolist())
        for i in range(SAMPLES_PER_BATCH):
            remove_b = i < (SAMPLES_PER_BATCH // 2)
            sample_id = f"{batch_name}_S{i:02d}"
            sample, counts = build_sample_confounded(pool, sample_id, batch_name, remove_b, target_celltypes)
            samples.append(sample)
            summary_rows.append({"sample_id": sample_id, "batch": batch_name, "remove_b": remove_b, **counts})
    pd.DataFrame(summary_rows).to_csv(f"{SAVE_DIR}/sample_composition.csv", index=False)
    return ad.concat(samples, join="outer", merge="first")


def main():
    adata = load_and_align()
    adata = annotate_cell_types_per_batch(adata)
    print("Cell type counts per batch:")
    print(adata.obs.groupby(["batch", "cell_type"]).size().unstack(fill_value=0))
    sim_adata = build_dataset_confounded(adata)
    sim_adata.write_h5ad(OUT_DATA_PATH)
    print("Simulated confounded dataset:", sim_adata)
    comp = sim_adata.obs.groupby(["batch", "cell_type"]).size().unstack(fill_value=0)
    comp_frac = comp.div(comp.sum(axis=1), axis=0)
    print("\nCell-type fraction per batch (should show NK confound, B independent of batch):")
    print(comp_frac)
    comp.to_csv(f"{SAVE_DIR}/celltype_counts_by_batch.csv")
    comp_frac.to_csv(f"{SAVE_DIR}/celltype_frac_by_batch.csv")


if __name__ == "__main__":
    main()
