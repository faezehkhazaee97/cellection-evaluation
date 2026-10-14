#!/usr/bin/env python3
"""Collapses the CELECTION runs to one line per task for Fig. 2 / Table S3.

For every $RESULTS_DIR/R2-q1/Results/CELECTION/<Task>/*metrics_all.csv, averages auROC, macroF1 and
macroAUPRC over the test rows of model 'CELECTION' (all seed x fold runs). The result, one row per task
folder with columns model, dataset, auROC_test, macroF1_test and macroAUPRC_test, goes to
celection_test_summary.csv in the same CELECTION folder.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

from pathlib import Path

import pandas as pd


BASE_DIR = Path(f"{RESULTS_DIR}/R2-q1/Results/CELECTION")
METRIC_GLOB = "*/*metrics_all.csv"
MODEL_NAME = "CELECTION"
OUTPUT_MODEL_NAME = "CELECTION"
SPLIT_NAME = "test"
OUTPUT_FILENAME = "celection_test_summary.csv"


def summarize_metrics(metrics_path: Path) -> dict[str, float | str]:
    """Mean test metrics of the CELECTION rows of one *metrics_all.csv; dataset = its folder name."""
    df = pd.read_csv(metrics_path)
    df = df[(df["model"] == MODEL_NAME) & (df["split"] == SPLIT_NAME)]
    if df.empty:
        raise ValueError(f"No {MODEL_NAME} {SPLIT_NAME} rows in {metrics_path}")
    return {
        "model": OUTPUT_MODEL_NAME,
        "dataset": metrics_path.parent.name,
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
