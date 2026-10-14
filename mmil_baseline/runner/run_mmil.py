#!/usr/bin/env python3
"""MMIL baseline end to end: input tables from $DATA_DIR, then the R model, one task or all three.

Tables for covid, lupus_disease and lupus_race (DATASET_CONFIG) follow the prep_*.py rules: same
donor and label columns, same per-donor subsampling with default_rng(0), all cells for covid and
10% for the lupus tasks unless --sample-frac overrides it for every selected task. Folds are always
the predefined benchmark splits (covid_5folds.csv, lupus_5folds.csv, race_5folds.csv). The matching
mmil_*.R script is then run as `Rscript <script> <cells.csv> <folds.csv> <output_dir>` from the
mmil_baseline folder.

    python runner/run_mmil.py [--dataset all|covid|lupus_disease|lupus_race] [--rscript PATH]
                              [--sample-frac F] [--prepare-only] [--dry-run]

Reads covid/covid_Ziegler_processed.h5ad, covid/covid_5folds.csv, lupus/lupus_processed.h5ad,
lupus/lupus_5folds.csv and lupus/race_5folds.csv under $DATA_DIR. Each task gets a folder under
$RESULTS_DIR/R1-q5/Results/ (Covid/, Lupus_Disease/, Lupus_Race/) holding cells.csv, folds.csv and
label_counts.csv from this script plus fold_<k>_preds.csv and MMIL_summary.csv from R. COVID lands
in Covid/ here, whereas prep_covid.py writes to Results/ itself. Rscript must be on the PATH
unless --rscript is given; runner/README.md has the options.
"""
from __future__ import annotations

# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# mmil_baseline/: holds the mmil_*.R scripts and is the working directory of the R runs.
R1_CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(DATA_DIR)
RESULTS_ROOT = Path(RESULTS_DIR) / "R1-q5" / "Results"

DATASET_CONFIG = {
    "covid": {
        "adata_path": DATA_ROOT / "covid" / "covid_Ziegler_processed.h5ad",
        "folds_path": DATA_ROOT / "covid" / "covid_5folds.csv",
        "r_script": R1_CODE_ROOT / "mmil_covid.R",
        "output_dir": RESULTS_ROOT / "Covid",
        "sample_frac": 1.0,
        "donor_col": "donor_id",
    },
    "lupus_disease": {
        "adata_path": DATA_ROOT / "lupus" / "lupus_processed.h5ad",
        "folds_path": DATA_ROOT / "lupus" / "lupus_5folds.csv",
        "r_script": R1_CODE_ROOT / "mmil_lupus_disease.R",
        "output_dir": RESULTS_ROOT / "Lupus_Disease",
        "sample_frac": 0.10,
        "donor_col": "ind_cov",
        "label_col": "disease_cov",
        "positive_label": "sle",
    },
    "lupus_race": {
        "adata_path": DATA_ROOT / "lupus" / "lupus_processed.h5ad",
        "folds_path": DATA_ROOT / "lupus" / "race_5folds.csv",
        "r_script": R1_CODE_ROOT / "mmil_lupus_race.R",
        "output_dir": RESULTS_ROOT / "Lupus_Race",
        "sample_frac": 0.10,
        "donor_col": "ind_cov",
        "label_col": "pop_cov",
        "positive_label": "asian",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMIL benchmarks from repo-relative data paths.")
    parser.add_argument("--dataset", default="all", choices=["all", *DATASET_CONFIG])
    parser.add_argument("--rscript", default="Rscript", help="Rscript executable to use")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Override per-dataset cell subsampling fraction used during MMIL input preparation",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build cells.csv/folds.csv without running the R model",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def datasets_to_run(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(DATASET_CONFIG)
    return [dataset_arg]


def infer_covid_labels(adata) -> pd.Series:
    """0/1 label per cell from the first usable obs column: bag_label, case_control == "case", or
    disease__ontology_label in {"COVID-19", "long COVID-19"}."""
    if "bag_label" in adata.obs.columns and not adata.obs["bag_label"].isna().all():
        return adata.obs["bag_label"].astype(int)
    if "case_control" in adata.obs.columns and not adata.obs["case_control"].isna().all():
        return (adata.obs["case_control"] == "case").astype(int)
    if "disease__ontology_label" in adata.obs.columns:
        return adata.obs["disease__ontology_label"].isin(["COVID-19", "long COVID-19"]).astype(int)
    raise ValueError(
        "No usable COVID label found in adata.obs; expected 'bag_label', 'case_control', or "
        "'disease__ontology_label'."
    )


def subsample_by_donor(adata, donor_col: str, sample_frac: float) -> object:
    """Keep `sample_frac` of each donor's cells; 1 returns `adata` unchanged."""
    if not (0 < sample_frac <= 1):
        raise ValueError("sample_frac must be in (0, 1].")
    if sample_frac >= 1:
        return adata

    # Same seed, donor order and draw as prep_lupus_*.py, so both routes keep the same cells.
    rng = np.random.default_rng(0)
    keep_idx = []
    for _, idx in adata.obs.groupby(donor_col).groups.items():
        idx_list = list(idx)
        n_keep = max(1, int(np.ceil(len(idx_list) * sample_frac)))
        keep_idx.extend(rng.choice(idx_list, size=n_keep, replace=False))
    return adata[keep_idx].copy()


def build_cells_and_folds(dataset_name: str, sample_frac_override: float | None) -> tuple[Path, Path, Path]:
    """Write cells.csv, folds.csv and label_counts.csv for one task.

    Returns (cells_path, folds_path, output_dir).
    """
    import anndata as ad  # imported here so that --dry-run does not need anndata

    config = DATASET_CONFIG[dataset_name]
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {config['adata_path']}")
    adata = ad.read_h5ad(config["adata_path"])
    donor_col = config["donor_col"]
    sample_frac = config["sample_frac"] if sample_frac_override is None else sample_frac_override
    adata = subsample_by_donor(adata, donor_col=donor_col, sample_frac=sample_frac)

    print(f"[INFO] Building MMIL inputs for {dataset_name}")
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    df = pd.DataFrame(X, columns=adata.var_names)
    df.insert(0, "cell_id", adata.obs_names)
    df.insert(1, "donor_id", adata.obs[donor_col].values)

    if dataset_name == "covid":
        label_values = infer_covid_labels(adata).to_numpy()
        label_report = pd.DataFrame({"label": label_values}).value_counts().to_frame("count")
    else:
        labels = adata.obs[config["label_col"]].astype(str).str.lower()
        positive_label = config["positive_label"]
        if positive_label not in labels.unique():
            raise ValueError(
                f"Positive label '{positive_label}' not found in {config['label_col']}: {sorted(labels.unique())}"
            )
        label_values = (labels == positive_label).astype(int).to_numpy()
        label_report = pd.DataFrame({"label": labels}).value_counts().to_frame("count")

    df["bag_label"] = label_values
    donor_label_nunique = df.groupby("donor_id")["bag_label"].nunique()
    if (donor_label_nunique > 1).any():
        bad_donors = donor_label_nunique[donor_label_nunique > 1].index.tolist()
        raise ValueError(f"Inconsistent labels within donors: {bad_donors[:10]}")

    cells_path = output_dir / "cells.csv"
    folds_path = output_dir / "folds.csv"
    label_report.to_csv(output_dir / "label_counts.csv")
    df.to_csv(cells_path, index=False)

    folds = pd.read_csv(config["folds_path"])
    if "samples" in folds.columns and "donor_id" not in folds.columns:
        folds = folds.rename(columns={"samples": "donor_id"})
    if "donor_id" not in folds.columns or "fold" not in folds.columns:
        raise ValueError(f"{config['folds_path'].name} must contain donor_id and fold columns.")
    folds.to_csv(folds_path, index=False)

    return cells_path, folds_path, output_dir


def run_dataset(
    dataset_name: str,
    rscript_bin: str,
    sample_frac_override: float | None,
    prepare_only: bool,
    dry_run: bool,
) -> None:
    """Build one task's inputs and run its R script; `dry_run` only prints the steps."""
    config = DATASET_CONFIG[dataset_name]
    output_dir = config["output_dir"]
    cells_path = output_dir / "cells.csv"
    folds_path = output_dir / "folds.csv"

    if dry_run:
        print(f"[PREP] {config['adata_path']} + {config['folds_path']} -> {output_dir}")
        if prepare_only:
            return
        cmd = [rscript_bin, str(config["r_script"]), str(cells_path), str(folds_path), str(output_dir)]
        print(f"[RUN] {' '.join(cmd)}")
        return

    cells_path, folds_path, output_dir = build_cells_and_folds(dataset_name, sample_frac_override)
    if prepare_only:
        return

    cmd = [rscript_bin, str(config["r_script"]), str(cells_path), str(folds_path), str(output_dir)]
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=R1_CODE_ROOT)


def main() -> None:
    args = parse_args()
    for dataset_name in datasets_to_run(args.dataset):
        run_dataset(
            dataset_name=dataset_name,
            rscript_bin=args.rscript,
            sample_frac_override=args.sample_frac,
            prepare_only=args.prepare_only,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
