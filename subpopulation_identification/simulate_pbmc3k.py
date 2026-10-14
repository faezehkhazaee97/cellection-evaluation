"""Simulated PBMC3k patients whose cases differ from controls only in B-cell abundance.

Real PBMC3k cells are filtered (genes and cells with fewer than 3 counts removed), library-size
normalised, log1p-transformed and clustered (Leiden at resolution 0.5, or KMeans with k=8 on the PCA
embedding when igraph/leidenalg is missing). Each cluster is called B, T, Mono or NK after whichever
single marker (MS4A1, CD3D, CD14, NKG7) has the highest mean log-normalised expression in it. These
cluster-level labels in obs["cell_type"] are the ground truth for everything downstream.

From these cells we draw 50 patients of 1,000 cells with replacement. P0-P24 are cases
(obs["disease"] = 1) with a B fraction from U(0.30, 0.40); P25-P49 are controls
(obs["disease"] = 0) with U(0.00, 0.05). The rest of each bag comes from one shared pool of non-B
cells, so T/Mono/NK proportions match in expectation, and expression profiles are left as they are.
The fractions come from a generator seeded with 0, but the cells are drawn with pandas sampling
without a fixed seed, so a rerun picks different cells. Assertions check bag sizes, the case/control
split and the realised B fractions before the sanity tables are written.

Reads $DATA_DIR/pbmc3k/pbmc3k_raw.h5ad (scanpy downloads it from backup_url if missing). Writes to
$DATA_DIR/pbmc3k/:
  pbmc3k_processed_with_celltypes.h5ad   all real cells, log-normalised (raw counts in
                                          layers["counts"]), with obs["leiden"] and obs["cell_type"]
  pbmc3k_simulated_patients_Bcell_causal.h5ad   the cohort (bag id in obs["patient_id"]), read by
                                                identify_subpopulations_pbmc3k.py
  pbmc3k_simulated_patients_sanity_by_patient.csv   patient_id, disease, b_fraction, n_cells
  pbmc3k_simulated_patients_sanity_summary.csv      one row: cohort size, case/control counts,
                                                    B-fraction range per group, total cells
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets; the cohort is written here too


import numpy as np
import pandas as pd
import scanpy as sc

SAVE_DIR = f"{DATA_DIR}/pbmc3k"
os.makedirs(SAVE_DIR, exist_ok=True)

pbmc3k_path = os.path.join(SAVE_DIR, "pbmc3k_raw.h5ad")

pbmc3k = sc.read(
    filename=pbmc3k_path,
    backup_url="http://falexwolf.de/data/pbmc3k_raw.h5ad"
)
sc.pp.filter_genes(pbmc3k, min_counts=3)
sc.pp.filter_cells(pbmc3k, min_counts=3)

pbmc3k.layers["counts"] = pbmc3k.X.copy()

sc.pp.normalize_total(pbmc3k)
sc.pp.log1p(pbmc3k)

sc.pp.pca(pbmc3k)
sc.pp.neighbors(pbmc3k)
try:
    sc.tl.leiden(pbmc3k, resolution=0.5)
except Exception as exc:
    msg = str(exc).lower()
    if ("igraph" not in msg) and ("leidenalg" not in msg):
        raise
    from sklearn.cluster import KMeans

    print("[WARN] Leiden unavailable (missing igraph/leidenalg). Falling back to KMeans clusters.")
    km = KMeans(n_clusters=8, random_state=0, n_init=20)
    pbmc3k.obs["leiden"] = km.fit_predict(pbmc3k.obsm["X_pca"]).astype(str)
sc.tl.umap(pbmc3k)

marker_genes = {
    "B": ["MS4A1"],
    "T": ["CD3D"],
    "Mono": ["CD14"],
    "NK": ["NKG7"],
}

# One label per cluster, not per cell: the type whose marker has the highest cluster mean.
cluster_means = pd.DataFrame(
    pbmc3k.X.toarray() if hasattr(pbmc3k.X, "toarray") else pbmc3k.X,
    columns=pbmc3k.var_names,
    index=pbmc3k.obs["leiden"]
).groupby(level=0).mean()

cluster_to_type = {}

for cluster in cluster_means.index:
    scores = {
        "B": cluster_means.loc[cluster, "MS4A1"],
        "T": cluster_means.loc[cluster, "CD3D"],
        "Mono": cluster_means.loc[cluster, "CD14"],
        "NK": cluster_means.loc[cluster, "NKG7"],
    }
    cluster_to_type[cluster] = max(scores, key=scores.get)

pbmc3k.obs["cell_type"] = pbmc3k.obs["leiden"].map(cluster_to_type)
pbmc3k.write(
    os.path.join(SAVE_DIR, "pbmc3k_processed_with_celltypes.h5ad")
)
print(pbmc3k.obs["cell_type"].value_counts())

N_PATIENTS = 50
CELLS_PER_PATIENT = 1000
CASE_FRAC = 0.5
CASE_B_FRAC_MIN = 0.30
CASE_B_FRAC_MAX = 0.40
CONTROL_B_FRAC_MAX = 0.05
rng = np.random.default_rng(0)
patients = []

# The first N_PATIENTS*CASE_FRAC bags are cases. Only the number of B cells differs between the
# groups; the non-B cells come from the same pool for every patient.
for i in range(N_PATIENTS):
    is_case = i < int(N_PATIENTS * CASE_FRAC)

    if is_case:
        b_frac = rng.uniform(0.30, 0.40)
    else:
        b_frac = rng.uniform(0.00, 0.05)

    # int() truncates, so realised case fractions lie in [0.30, 0.40), as the checks below expect.
    n_b = int(CELLS_PER_PATIENT * b_frac)
    n_rest = CELLS_PER_PATIENT - n_b

    b_cells = pbmc3k[pbmc3k.obs["cell_type"] == "B"].obs.sample(n_b, replace=True)
    other_cells = pbmc3k[pbmc3k.obs["cell_type"] != "B"].obs.sample(n_rest, replace=True)

    patient_cells = pd.concat([b_cells, other_cells])
    patient_cells["patient_id"] = f"P{i}"
    patient_cells["disease"] = int(is_case)

    patients.append(patient_cells)
sim_obs = pd.concat(patients)
# sim_obs.index holds the sampled cell names, with repeats: pull those rows from the real data,
# attach patient_id/disease, then make the repeated names unique.
sim_adata = pbmc3k[sim_obs.index].copy()
sim_adata.obs = sim_obs
sim_adata.obs_names_make_unique()
sim_adata.write(
    os.path.join(SAVE_DIR, "pbmc3k_simulated_patients_Bcell_causal.h5ad")
)

# Sanity checks: assertions first, then the tables.
patient_sizes = sim_adata.obs.groupby("patient_id", observed=False).size()
assert (patient_sizes == CELLS_PER_PATIENT).all(), (
    f"Patient size mismatch detected. Expected {CELLS_PER_PATIENT} cells per patient."
)

disease_per_patient = sim_adata.obs.groupby("patient_id", observed=False)["disease"].first()
expected_case = int(N_PATIENTS * CASE_FRAC)
expected_control = N_PATIENTS - expected_case
disease_counts = disease_per_patient.value_counts().to_dict()
assert disease_counts.get(1, 0) == expected_case and disease_counts.get(0, 0) == expected_control, (
    f"Disease split mismatch. Expected case/control={expected_case}/{expected_control}, "
    f"got {disease_counts.get(1, 0)}/{disease_counts.get(0, 0)}."
)

patient_b_frac = sim_adata.obs.groupby("patient_id", observed=False)["cell_type"].apply(
    lambda s: (s == "B").mean()
)
case_b = patient_b_frac[disease_per_patient == 1]
control_b = patient_b_frac[disease_per_patient == 0]

assert case_b.min() >= CASE_B_FRAC_MIN and case_b.max() < CASE_B_FRAC_MAX, (
    f"Case B fraction out of range: [{case_b.min():.3f}, {case_b.max():.3f}]"
)
assert control_b.min() >= 0.0 and control_b.max() <= CONTROL_B_FRAC_MAX, (
    f"Control B fraction out of range: [{control_b.min():.3f}, {control_b.max():.3f}]"
)

sanity_by_patient = pd.DataFrame(
    {
        "patient_id": patient_b_frac.index,
        "disease": disease_per_patient.reindex(patient_b_frac.index).values,
        "b_fraction": patient_b_frac.values,
        "n_cells": patient_sizes.reindex(patient_b_frac.index).values,
    }
).sort_values("patient_id")
sanity_by_patient.to_csv(
    os.path.join(SAVE_DIR, "pbmc3k_simulated_patients_sanity_by_patient.csv"),
    index=False,
)

sanity_summary = pd.DataFrame(
    [
        {
            "n_patients": N_PATIENTS,
            "cells_per_patient": CELLS_PER_PATIENT,
            "n_case_patients": int((disease_per_patient == 1).sum()),
            "n_control_patients": int((disease_per_patient == 0).sum()),
            "case_b_min": float(case_b.min()),
            "case_b_max": float(case_b.max()),
            "control_b_min": float(control_b.min()),
            "control_b_max": float(control_b.max()),
            "total_cells": sim_adata.n_obs,
        }
    ]
)
sanity_summary.to_csv(
    os.path.join(SAVE_DIR, "pbmc3k_simulated_patients_sanity_summary.csv"),
    index=False,
)

print("\n=== SIMULATION SANITY CHECKS PASSED ===")
print(sanity_summary.to_string(index=False))
