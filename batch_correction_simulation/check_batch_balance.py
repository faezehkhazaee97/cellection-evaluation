#!/usr/bin/env python3
"""Is the SEA-AD batch covariate balanced across AD and control donors?

seaad_with_batch.py conditions on sex; if sex tracked disease, the covariate would carry label
information. This prints the donor-level batch x disease crosstab with row and column
percentages, the chi-square p-value and Cramer's V. Donors, not cells, are counted so that
donors with many cells do not dominate.

Cells with a null --celltype-key (default "Supertype") are dropped, as in the SEA-AD scripts;
each donor (--sample-key, default "donor_id") then gives one row with its --task-key
("disease") and --batch-key ("sex").
    --cell-level  also print the table over cells.
    --split       shuffle donors with np.random.RandomState(--seed, default 10), put the first
                  --train-frac (default 0.8) in training and print the tables for training and
                  held-out donors separately, mimicking the seeded 80/20 donor split of
                  seaad_with_batch.py / seaad_without_batch.py.

Reads --data-path (default $DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad); stdout only.
"""

# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets

import argparse

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import chi2_contingency


def build_donor_df(adata, sample_key, task_key, batch_key):
    """One row per donor (its first cell), after dropping cells with a missing sample, task or batch."""
    cols = [sample_key, task_key, batch_key]
    donor_df = (
        adata.obs[cols]
        .dropna()
        .drop_duplicates(subset=[sample_key])
    )
    return donor_df


def print_balance(ct, label):
    """Print the crosstab, its row and column percentages, chi-square p-value and Cramer's V."""
    print(f"\n{label} counts:")
    print(ct)
    print("\nRow %:")
    print(ct.div(ct.sum(axis=1), axis=0).round(3))
    print("\nCol %:")
    print(ct.div(ct.sum(axis=0), axis=1).round(3))

    if min(ct.shape) < 2:
        print("\nChi-square not defined (only one level in a dimension).")
        return

    chi2, p, dof, expected = chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    print(f"\nChi2 p-value = {p:.4g}, Cramer's V = {cramers_v:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Check batch covariate vs disease label balance."
    )
    parser.add_argument(
        "--data-path",
        default=f"{DATA_DIR}/sea-ad/MTG_glia_intersection_norm.h5ad",
        help="Path to h5ad file.",
    )
    parser.add_argument("--sample-key", default="donor_id")
    parser.add_argument("--task-key", default="disease")
    parser.add_argument("--batch-key", default="sex")
    parser.add_argument(
        "--celltype-key",
        default="Supertype",
        help="Filter to non-null cell type, matching main scripts.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Also report train/val balance after donor split.",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument(
        "--cell-level",
        action="store_true",
        help="Also print cell-level table (not recommended for balance tests).",
    )
    args = parser.parse_args()

    adata = sc.read_h5ad(args.data_path)
    adata = adata[adata.obs[args.celltype_key].notna()].copy()

    donor_df = build_donor_df(adata, args.sample_key, args.task_key, args.batch_key)
    ct = pd.crosstab(donor_df[args.batch_key], donor_df[args.task_key])
    print_balance(ct, "Donor-level batch x disease")

    if args.cell_level:
        ct_cells = pd.crosstab(adata.obs[args.batch_key], adata.obs[args.task_key])
        print_balance(ct_cells, "Cell-level batch x disease")

    if args.split:
        donors = donor_df[args.sample_key].to_numpy()
        rng = np.random.RandomState(args.seed)
        rng.shuffle(donors)
        n_train = int(args.train_frac * len(donors))
        train_donors = set(donors[:n_train])

        donor_df_train = donor_df[donor_df[args.sample_key].isin(train_donors)]
        donor_df_val = donor_df[~donor_df[args.sample_key].isin(train_donors)]

        ct_train = pd.crosstab(donor_df_train[args.batch_key], donor_df_train[args.task_key])
        ct_val = pd.crosstab(donor_df_val[args.batch_key], donor_df_val[args.task_key])
        print_balance(ct_train, "TRAIN donor-level batch x disease")
        print_balance(ct_val, "VAL donor-level batch x disease")


if __name__ == "__main__":
    main()
