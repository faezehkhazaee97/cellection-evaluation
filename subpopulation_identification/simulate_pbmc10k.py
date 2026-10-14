"""PBMC10k version of the simulated B-cell cohort from simulate_pbmc3k.py.

Filtering, clustering, the marker-based B/T/Mono/NK labels and the sampling scheme are those of
simulate_pbmc3k.py: case B fraction ~ U(0.30, 0.40), control ~ U(0.00, 0.05), fractions from a
generator seeded with 0, cells drawn with replacement by pandas without a fixed seed, and all non-B
cells from one shared pool, so B-cell abundance is the only difference between the groups.
Here there are 40 patients of 500 cells: P0-P19 are cases (obs["disease"] = 1), P20-P39 controls
(obs["disease"] = 0), and the bag id is obs["patient_id"].
Unlike the 3k script, only the four marker genes are densified for the cluster means, and the
script stops if a marker gene is missing or the B or non-B pool is empty. The same assertions run
before the sanity tables are written.

Reads $DATA_DIR/pbmc10k/pbmc10k_raw.h5ad, the raw counts of the 10x Genomics PBMC 10k (v3) filtered
feature-barcode matrix saved as h5ad. It is not downloaded; the script stops if the file is missing.
Writes to $DATA_DIR/pbmc10k/:
  pbmc10k_processed_with_celltypes.h5ad   all real cells, log-normalised (raw counts in
                                           layers["counts"]), with obs["leiden"] and obs["cell_type"]
  pbmc10k_simulated_patients_Bcell_causal.h5ad   the cohort; input of identify_subpopulations_pbmc10k.py
  pbmc10k_simulated_patients_sanity_by_patient.csv   patient_id, disease, b_fraction, n_cells
  pbmc10k_simulated_patients_sanity_summary.csv      one-row cohort summary
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets; the cohort is written here too


import numpy as np
import pandas as pd
import scanpy as sc

SAVE_DIR = f"{DATA_DIR}/pbmc10k"
os.makedirs(SAVE_DIR, exist_ok=True)

pbmc10k_path = os.path.join(SAVE_DIR, "pbmc10k_raw.h5ad")
if not os.path.exists(pbmc10k_path):
    raise FileNotFoundError(
        f"Missing input file: {pbmc10k_path}\n"
        "Place PBMC 10k as pbmc10k_raw.h5ad in SAVE_DIR and rerun."
    )

pbmc = sc.read_h5ad(pbmc10k_path)

sc.pp.filter_genes(pbmc, min_counts=3)
sc.pp.filter_cells(pbmc, min_counts=3)

pbmc.layers["counts"] = pbmc.X.copy()

sc.pp.normalize_total(pbmc)
sc.pp.log1p(pbmc)

sc.pp.pca(pbmc)
sc.pp.neighbors(pbmc)
try:
    sc.tl.leiden(pbmc, resolution=0.5)
except Exception as exc:
    msg = str(exc).lower()
    if ("igraph" not in msg) and ("leidenalg" not in msg):
        raise
    from sklearn.cluster import KMeans

    print("[WARN] Leiden unavailable (missing igraph/leidenalg). Falling back to KMeans clusters.")
    km = KMeans(n_clusters=8, random_state=0, n_init=20)
    pbmc.obs["leiden"] = km.fit_predict(pbmc.obsm["X_pca"]).astype(str)

sc.tl.umap(pbmc)

marker_genes = {
    "B": "MS4A1",
    "T": "CD3D",
    "Mono": "CD14",
    "NK": "NKG7",
}

missing_markers = [g for g in marker_genes.values() if g not in pbmc.var_names]
if missing_markers:
    raise ValueError(f"Missing marker genes in var_names: {missing_markers}")

marker_list = list(marker_genes.values())
marker_idx = [pbmc.var_names.get_loc(g) for g in marker_list]
X_marker = pbmc[:, marker_idx].X
X_marker = X_marker.toarray() if hasattr(X_marker, "toarray") else np.asarray(X_marker)

# One label per cluster, not per cell: the type whose marker has the highest cluster mean.
cluster_means = (
    pd.DataFrame(X_marker, columns=marker_list, index=pbmc.obs["leiden"])
    .groupby(level=0, observed=False)
    .mean()
)

cluster_to_type = {}
for cluster in cluster_means.index:
    scores = {
        "B": cluster_means.loc[cluster, marker_genes["B"]],
        "T": cluster_means.loc[cluster, marker_genes["T"]],
        "Mono": cluster_means.loc[cluster, marker_genes["Mono"]],
        "NK": cluster_means.loc[cluster, marker_genes["NK"]],
    }
    cluster_to_type[cluster] = max(scores, key=scores.get)

pbmc.obs["cell_type"] = pbmc.obs["leiden"].map(cluster_to_type)
pbmc.write(os.path.join(SAVE_DIR, "pbmc10k_processed_with_celltypes.h5ad"))
print(pbmc.obs["cell_type"].value_counts())

N_PATIENTS = 40
CELLS_PER_PATIENT = 500
CASE_FRAC = 0.5
CASE_B_FRAC_MIN = 0.30
CASE_B_FRAC_MAX = 0.40
CONTROL_B_FRAC_MAX = 0.05
rng = np.random.default_rng(0)
patients = []

b_pool = pbmc[pbmc.obs["cell_type"] == "B"].obs
non_b_pool = pbmc[pbmc.obs["cell_type"] != "B"].obs
if b_pool.shape[0] == 0:
    raise ValueError("No B cells found after annotation; cannot simulate B-cell causal data.")
if non_b_pool.shape[0] == 0:
    raise ValueError("No non-B cells found after annotation; cannot simulate mixed patient bags.")

# The first N_PATIENTS*CASE_FRAC bags are cases. Only the number of B cells differs between the
# groups; the non-B cells come from the same pool for every patient.
for i in range(N_PATIENTS):
    is_case = i < int(N_PATIENTS * CASE_FRAC)

    if is_case:
        b_frac = rng.uniform(CASE_B_FRAC_MIN, CASE_B_FRAC_MAX)
    else:
        b_frac = rng.uniform(0.00, CONTROL_B_FRAC_MAX)

    # int() truncates, so realised case fractions lie in [0.30, 0.40), as the checks below expect.
    n_b = int(CELLS_PER_PATIENT * b_frac)
    n_rest = CELLS_PER_PATIENT - n_b

    b_cells = b_pool.sample(n_b, replace=True)
    other_cells = non_b_pool.sample(n_rest, replace=True)

    patient_cells = pd.concat([b_cells, other_cells])
    patient_cells["patient_id"] = f"P{i}"
    patient_cells["disease"] = int(is_case)
    patients.append(patient_cells)

sim_obs = pd.concat(patients)
# sim_obs.index holds the sampled cell names, with repeats: pull those rows from the real data,
# attach patient_id/disease, then make the repeated names unique.
sim_adata = pbmc[sim_obs.index].copy()
sim_adata.obs = sim_obs
sim_adata.obs_names_make_unique()
sim_adata.write(os.path.join(SAVE_DIR, "pbmc10k_simulated_patients_Bcell_causal.h5ad"))

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
    os.path.join(SAVE_DIR, "pbmc10k_simulated_patients_sanity_by_patient.csv"),
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
    os.path.join(SAVE_DIR, "pbmc10k_simulated_patients_sanity_summary.csv"),
    index=False,
)

print("\n=== SIMULATION SANITY CHECKS PASSED ===")
print(sanity_summary.to_string(index=False))
