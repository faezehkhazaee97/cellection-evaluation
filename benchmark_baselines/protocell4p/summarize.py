#!/usr/bin/env python3
"""Test-set averages for every ProtoCell4P task and prototype count (Fig. 2 / Table S3).

Globs $RESULTS_DIR/R2-q1/Results/4_ProtoCell4P/<Task>/proto_<n>/*metrics_all.csv, averages auROC, macroF1
and macroAUPRC over the test rows of each file, and writes protocell4p_test_summary.csv to the
4_ProtoCell4P folder: one row per task and prototype count, with the proto_<n> folder name in the model
column and the task folder in dataset.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

from pathlib import Path

import pandas as pd


BASE_DIR = Path(f"{RESULTS_DIR}/R2-q1/Results/4_ProtoCell4P")
METRIC_GLOB = "*/*/*metrics_all.csv"
SPLIT_NAME = "test"
OUTPUT_FILENAME = "protocell4p_test_summary.csv"


def summarize_metrics(metrics_path: Path) -> dict[str, float | str]:
    """Mean test metrics of one file, labelled by its proto_<n> folder (model) and task folder (dataset)."""
    df = pd.read_csv(metrics_path)
    df = df[df["split"] == SPLIT_NAME]
    if df.empty:
        raise ValueError(f"No {SPLIT_NAME} rows in {metrics_path}")
    proto_name = metrics_path.parent.name
    dataset_name = metrics_path.parent.parent.name
    return {
        "model": proto_name,
        "dataset": dataset_name,
        "auROC_test": df["auROC"].mean(),
        "macroF1_test": df["macroF1"].mean(),
        "macroAUPRC_test": df["macroAUPRC"].mean(),
    }


def main() -> None:
    """Summarise every metrics file under BASE_DIR and write OUTPUT_FILENAME there."""
    metrics_files = sorted(BASE_DIR.glob(METRIC_GLOB))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics files found under {BASE_DIR}")

    rows = [summarize_metrics(metrics_path) for metrics_path in metrics_files]
    out_path = BASE_DIR / OUTPUT_FILENAME
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
