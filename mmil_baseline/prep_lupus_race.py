"""Lupus ancestry task (Asian vs White): prep_lupus_disease.py with a different label.

The 10% per-donor subsample (SAMPLE_FRAC = 0.10, default_rng(RNG_SEED = 0)) keeps the same cells
as the disease task. Here bag_label is 1 when obs["pop_cov"], lower-cased, is "asian", else 0, and
the folds come from race_5folds.csv. Stops if "asian" is missing from pop_cov or a donor carries
two labels.

Reads $DATA_DIR/lupus/lupus_processed.h5ad and race_5folds.csv. Writes to
$RESULTS_DIR/R1-q5/Results/Lupus_Race/: cells.csv, folds.csv (with a donor_id column), and
label_counts.csv (kept cells per lower-cased pop_cov value). Next step:
Rscript mmil_lupus_race.R <cells.csv> <folds file> <output_dir>.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import numpy as np
import pandas as pd
import scanpy as sc

adata_path = f"{DATA_DIR}/lupus/lupus_processed.h5ad"
folds_path = f"{DATA_DIR}/lupus/race_5folds.csv"
out_dir = f"{RESULTS_DIR}/R1-q5/Results/Lupus_Race"
os.makedirs(out_dir, exist_ok=True)
print("Loading AnnData...")
adata = sc.read_h5ad(adata_path)
donor_col = "ind_cov"
label_col = "pop_cov"
pos_label = "asian"
SAMPLE_FRAC = 0.10
RNG_SEED = 0
if not (0 < SAMPLE_FRAC <= 1):
    raise ValueError("SAMPLE_FRAC must be in (0, 1].")
if SAMPLE_FRAC < 1:
    # One RNG stream across donors in groupby order: same draw as prep_lupus_disease.py.
    rng = np.random.default_rng(RNG_SEED)
    keep_idx = []
    for donor, idx in adata.obs.groupby(donor_col).groups.items():
        idx_list = list(idx)
        n_keep = max(1, int(np.ceil(len(idx_list) * SAMPLE_FRAC)))
        keep = rng.choice(idx_list, size=n_keep, replace=False)
        keep_idx.extend(keep)
    adata = adata[keep_idx].copy()
print("Creating cells.csv...")
X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
df = pd.DataFrame(X, columns=adata.var_names)
df.insert(0, "cell_id", adata.obs_names)
df.insert(1, "donor_id", adata.obs[donor_col].values)
labels = adata.obs[label_col].astype(str).str.lower()
if pos_label not in labels.unique():
    raise ValueError(
        f"pos_label '{pos_label}' not found in {label_col}: "
        f"{sorted(labels.unique())}"
    )
label_values = (labels == pos_label).astype(int).to_numpy()
df["bag_label"] = label_values

label_nunique = df.groupby("donor_id")["bag_label"].nunique()
if (label_nunique > 1).any():
    bad_donors = label_nunique[label_nunique > 1].index.tolist()
    raise ValueError(
        f"Inconsistent labels within donors: {bad_donors[:10]}"
    )
pd.DataFrame({"label": labels}).value_counts().to_frame("count").to_csv(
    f"{out_dir}/label_counts.csv"
)
df.to_csv(f"{out_dir}/cells.csv", index=False)
print(f"Saved: {out_dir}/cells.csv")
print("Using predefined 5-fold split...")
folds = pd.read_csv(folds_path)
# The benchmark fold files call the donor column "samples"; the R scripts expect "donor_id".
if "samples" in folds.columns and "donor_id" not in folds.columns:
    folds = folds.rename(columns={"samples": "donor_id"})
if "donor_id" not in folds.columns or "fold" not in folds.columns:
    raise ValueError("race_5folds.csv must contain donor_id and fold columns.")
folds.to_csv(f"{out_dir}/folds.csv", index=False)
print(f"Saved: {out_dir}/folds.csv")
