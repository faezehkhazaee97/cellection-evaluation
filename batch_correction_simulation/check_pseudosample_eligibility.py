"""How many pancreas samples survive the with-beta / no-beta pseudo-sample split?

make_pseudo_labels() in pancreas_pseudosamples.py (and the _lab_counts copy) drops any sample
without at least one beta and one non-beta cell. This prints, per study (obs["dataset"] in
$DATA_DIR/pancreatic_w_batch/Fig2_obj.h5ad), how many samples there are and how many qualify,
then the (n_beta, n_nonbeta, valid) combinations for segerstolpe and xin. Sample ID = barcode
prefix before the first ".". Stdout only.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets

import scanpy as sc

DATA_PATH = f"{DATA_DIR}/pancreatic_w_batch/Fig2_obj.h5ad"

batch_key = "dataset"
cell_type_key = "celltype"
beta_label = "beta"

adata = sc.read_h5ad(DATA_PATH)

obs = adata.obs.copy()

# same sample ID rule as make_pseudo_labels()
obs["bag_id"] = obs.index.to_series().str.split(".", n=1).str[0]

obs["is_beta"] = (obs[cell_type_key] == beta_label)

# observed=True: only (study, sample) pairs that occur, not every category combination
per_bag = (
    obs
    .groupby([batch_key, "bag_id"], observed=True)["is_beta"]
    .agg(
        n_beta="sum",
        n_total="count"
    )
)

per_bag["n_nonbeta"] = per_bag["n_total"] - per_bag["n_beta"]

# the rule make_pseudo_labels() applies
per_bag["valid"] = (
    (per_bag["n_beta"] > 0) &
    (per_bag["n_nonbeta"] > 0)
)

summary = (
    per_bag
    .reset_index()
    .groupby(batch_key)["valid"]
    .agg(
        n_samples="size",
        n_valid="sum",
        n_invalid=lambda x: (~x).sum()
    )
)

print("\n=== Per-dataset sample validity summary (corrected) ===")
print(summary)

for ds in ["segerstolpe", "xin"]:
    print(f"\n=== {ds.upper()} bag composition ===")
    sub = per_bag.loc[ds]
    print(sub[["n_beta", "n_nonbeta", "valid"]].value_counts())