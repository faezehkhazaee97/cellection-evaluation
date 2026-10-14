"""Stratified 5-fold donor split for the lupus SLE-vs-healthy MMIL run, beside the predefined folds.

Donors are dealt into folds 1-5 by StratifiedKFold(n_splits=5, shuffle=True, random_state=0), so
each fold keeps the SLE/healthy donor ratio; the fold x label donor counts are printed as a check.
Pass the output to mmil_lupus_disease.R as its folds argument.

Reads $RESULTS_DIR/R1-q5/Results/Lupus_Disease/cells.csv from prep_lupus_disease.py (only donor_id
and bag_label are used) and writes folds_stratified.csv (donor_id, fold) to the same folder.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import pandas as pd
from sklearn.model_selection import StratifiedKFold

cells_path = f"{RESULTS_DIR}/R1-q5/Results/Lupus_Disease/cells.csv"
out_path = f"{RESULTS_DIR}/R1-q5/Results/Lupus_Disease/folds_stratified.csv"

cells = pd.read_csv(cells_path)
donor_labels = cells[["donor_id","bag_label"]].drop_duplicates()
donor_labels["bag_label"] = donor_labels["bag_label"].astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
folds = []
for fold, (_, test_idx) in enumerate(skf.split(donor_labels["donor_id"], donor_labels["bag_label"]), start=1):
    folds.append(pd.DataFrame({"donor_id": donor_labels.iloc[test_idx]["donor_id"], "fold": fold}))

folds_df = pd.concat(folds, ignore_index=True)
folds_df.to_csv(out_path, index=False)

merged = donor_labels.merge(folds_df, on="donor_id")
print(pd.crosstab(merged["fold"], merged["bag_label"]))
