#!/usr/bin/env python3
"""One row per task for the DeepSet results of Fig. 2 / Table S3.

Only the raw-gene runs count: from each $RESULTS_DIR/R2-q1/Results/2_DeepSet/<Task>/*metrics_all.csv the
rows with model 'DeepSet' and split 'test' are averaged, and DeepSet_PC is ignored. A task folder with no
metrics file falls back to its *results_all_folds.csv files instead, one summary row per file (the PC_
file included). The table goes to deepset_test_summary.csv in the 2_DeepSet folder, with columns model,
dataset, auROC_test, macroF1_test and macroAUPRC_test.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

from pathlib import Path

import pandas as pd


BASE_DIR = Path(f"{RESULTS_DIR}/R2-q1/Results/2_DeepSet")
METRIC_GLOB = "*/*metrics_all.csv"
FALLBACK_GLOB = "*/*results_all_folds.csv"
MODEL_NAME = "DeepSet"
SPLIT_NAME = "test"
OUTPUT_FILENAME = "deepset_test_summary.csv"


def summarize_metrics(metrics_path: Path) -> dict[str, float | str]:
    """Mean test metrics of the DeepSet rows of one *metrics_all.csv; dataset = its folder name."""
    df = pd.read_csv(metrics_path)
    df = df[(df["model"] == MODEL_NAME) & (df["split"] == SPLIT_NAME)]
    if df.empty:
        raise ValueError(f"No {MODEL_NAME} {SPLIT_NAME} rows in {metrics_path}")
    return {
        "model": MODEL_NAME,
        "dataset": metrics_path.parent.name,
        "auROC_test": df["auROC"].mean(),
        "macroF1_test": df["macroF1"].mean(),
        "macroAUPRC_test": df["macroAUPRC"].mean(),
    }


def summarize_fallback(results_path: Path) -> dict[str, float | str]:
    """Column means of one *results_all_folds.csv (one row per run); dataset = its folder name."""
    df = pd.read_csv(results_path)
    if df.empty:
        raise ValueError(f"Empty results file: {results_path}")
    return {
        "model": MODEL_NAME,
        "dataset": results_path.parent.name,
        "auROC_test": df["auROC_test"].mean(),
        "macroF1_test": df["macroF1_test"].mean(),
        "macroAUPRC_test": df["macroAUPRC_test"].mean(),
    }


def main() -> None:
    """Write OUTPUT_FILENAME; *results_all_folds.csv only counts for folders without *metrics_all.csv."""
    metrics_files = sorted(BASE_DIR.glob(METRIC_GLOB))
    fallback_files = sorted(BASE_DIR.glob(FALLBACK_GLOB))
    if not metrics_files and not fallback_files:
        raise FileNotFoundError(f"No metrics or results files found under {BASE_DIR}")

    rows = [summarize_metrics(metrics_path) for metrics_path in metrics_files]
    fallback_dirs = {p.parent for p in metrics_files}
    for results_path in fallback_files:
        if results_path.parent in fallback_dirs:
            continue
        rows.append(summarize_fallback(results_path))
    out_path = BASE_DIR / OUTPUT_FILENAME
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
