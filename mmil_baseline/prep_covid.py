"""MMIL inputs for the Ziegler COVID-19 task: a dense per-cell table and the donor fold file.

Unlike the lupus tasks, every cell is kept. A row is cell_id, donor_id (obs["donor_id"]), every gene
of adata.X densified, and a 0/1 bag_label from the first usable source: obs["bag_label"], else
obs["case_control"] == "case", else obs["disease__ontology_label"] in {"COVID-19", "long COVID-19"}.
The script stops if a donor ends up with two labels. The predefined fold table is copied with its
"samples" column renamed to "donor_id" when needed.

Reads $DATA_DIR/covid/covid_Ziegler_processed.h5ad and covid_5folds.csv; writes cells.csv and
folds.csv to $RESULTS_DIR/R1-q5/Results/ (created if missing). Next step:
Rscript mmil_covid.R <cells.csv> <folds.csv> <output_dir>.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import pandas as pd
import scanpy as sc

adata_path = f"{DATA_DIR}/covid/covid_Ziegler_processed.h5ad"
existing_folds_path = (
    f"{DATA_DIR}/covid/covid_5folds.csv"
)
out_dir = f"{RESULTS_DIR}/R1-q5/Results"
os.makedirs(out_dir, exist_ok=True)
print("Loading AnnData...")
adata = sc.read_h5ad(adata_path)
print("Creating cells.csv...")
# .toarray() assumes a sparse X; a dense X fails here.
df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names)
df.insert(0, "cell_id", adata.obs_names)
df.insert(1, "donor_id", adata.obs["donor_id"].values)
label_series = None
if "bag_label" in adata.obs.columns and not adata.obs["bag_label"].isna().all():
    label_series = adata.obs["bag_label"]
elif "case_control" in adata.obs.columns and not adata.obs["case_control"].isna().all():
    label_series = (adata.obs["case_control"] == "case").astype(int)
elif "disease__ontology_label" in adata.obs.columns:
    # COVID-19 and long COVID-19 both count as cases.
    label_series = adata.obs["disease__ontology_label"].isin(
        ["COVID-19", "long COVID-19"]
    ).astype(int)
if label_series is None:
    raise ValueError(
        "No usable label found in adata.obs (expected non-empty 'bag_label', "
        "'case_control', or 'disease__ontology_label')."
    )
label_values = label_series.astype(int).to_numpy()
df["bag_label"] = label_values
print("bag_label counts:\n", df["bag_label"].value_counts())
label_nunique = df.groupby("donor_id")["bag_label"].nunique()
if (label_nunique > 1).any():
    bad_donors = label_nunique[label_nunique > 1].index.tolist()
    raise ValueError(
        f"Inconsistent labels within donors: {bad_donors[:10]}"
    )
df.to_csv(f"{out_dir}/cells.csv", index=False)
print(f"Saved: {out_dir}/cells.csv")
folds = pd.read_csv(existing_folds_path)
# The benchmark fold files call the donor column "samples"; the R scripts expect "donor_id".
if "samples" in folds.columns and "donor_id" not in folds.columns:
    folds = folds.rename(columns={"samples": "donor_id"})
if "donor_id" not in folds.columns or "fold" not in folds.columns:
    raise ValueError("covid_5folds.csv must contain donor_id and fold columns.")
folds.to_csv(f"{out_dir}/folds.csv", index=False)
print(f"Saved: {out_dir}/folds.csv")

