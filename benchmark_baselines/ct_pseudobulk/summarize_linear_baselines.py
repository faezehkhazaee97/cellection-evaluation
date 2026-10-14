#!/usr/bin/env python3
"""Logistic-regression summaries of the three linear baselines (ct_pb, proportion, pseudobulk) for Figure 2.

Reads the task folders written by ct_pseudobulk/*.py, ct_proportion/*.py and pseudobulk/*.py:
$RESULTS_DIR/R2-q1/Results/{ct_pb,proportion,pseudobulk}/<Dataset>/ (Covid, Lupus_Disease, Lupus_Race, Cardio,
CardioBiosample). In each of the three base folders it writes
  <Dataset>/logreg_test_summary.csv        mean and population std (ddof=0) of the LogisticRegression test auROC,
                                           macroF1 and macroAUPRC in *metrics_all.csv (columns dataset, n,
                                           <metric>_mean, <metric>_std); ValueError if there are no such rows
  <base>_logreg_results_mean_summary.csv   one row per dataset from LogisticRegression*_results_mean.csv (model,
                                           dataset, auROC_test, macroF1_test, macroAUPRC_test)
The *_logreg_results_mean_summary.csv files are the Figure 2 inputs. Their model column reads 'ct_pb' in all
three; plot/combine_all_test_summaries.py swaps in the baseline name. Random-forest results are not summarised.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

from pathlib import Path

import pandas as pd


BASE_DIRS = [
    Path(f"{RESULTS_DIR}/R2-q1/Results/ct_pb"),
    Path(f"{RESULTS_DIR}/R2-q1/Results/proportion"),
    Path(f"{RESULTS_DIR}/R2-q1/Results/pseudobulk"),
]
METRIC_GLOB = "*/*metrics_all.csv"
MEAN_GLOB = "*/LogisticRegression*_results_mean.csv"
MODEL_NAME = "LogisticRegression"
SUMMARY_MODEL_NAME = "ct_pb"
SPLIT_NAME = "test"
COMBINED_MEAN_FILENAME = "ct_pb_logreg_results_mean_summary.csv"


def summarize_metrics(metrics_path: Path) -> pd.DataFrame:
    """One-row table: mean and population std of the LogisticRegression test metrics in one *metrics_all.csv."""
    df = pd.read_csv(metrics_path)
    df = df[(df["model"] == MODEL_NAME) & (df["split"] == SPLIT_NAME)]
    if df.empty:
        raise ValueError(f"No {MODEL_NAME} {SPLIT_NAME} rows in {metrics_path}")
    return pd.DataFrame(
        [
            {
                "dataset": metrics_path.parent.name,
                "n": len(df),
                "auROC_mean": df["auROC"].mean(),
                "auROC_std": df["auROC"].std(ddof=0),
                "macroF1_mean": df["macroF1"].mean(),
                "macroF1_std": df["macroF1"].std(ddof=0),
                "macroAUPRC_mean": df["macroAUPRC"].mean(),
                "macroAUPRC_std": df["macroAUPRC"].std(ddof=0),
            }
        ]
    )


def summarize_mean_files(mean_files: list[Path]) -> pd.DataFrame:
    """One row per *_results_mean.csv (metric name -> value file), labelled with its dataset folder."""
    rows = []
    for mean_path in mean_files:
        df = pd.read_csv(mean_path)
        if df.empty:
            continue
        metric_col = df.columns[0]
        value_col = df.columns[1]
        metrics = dict(zip(df[metric_col].astype(str), df[value_col]))
        rows.append(
            {
                "model": SUMMARY_MODEL_NAME,
                "dataset": mean_path.parent.name,
                "auROC_test": metrics.get("auROC_test"),
                "macroF1_test": metrics.get("macroF1_test"),
                "macroAUPRC_test": metrics.get("macroAUPRC_test"),
            }
        )
    return pd.DataFrame(rows)


def summarize_base_dir(base_dir: Path) -> None:
    """Write the per-dataset test summaries and the combined mean-file summary for one baseline folder."""
    metrics_files = sorted(base_dir.glob(METRIC_GLOB))
    if not metrics_files:
        print(f"No metrics files found under {base_dir}")
        return

    for metrics_path in metrics_files:
        summary = summarize_metrics(metrics_path)
        out_path = metrics_path.parent / "logreg_test_summary.csv"
        summary.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    mean_files = sorted(base_dir.glob(MEAN_GLOB))
    if not mean_files:
        print(f"No mean files found under {base_dir}")
        return
    combined = summarize_mean_files(mean_files)
    combined_out = base_dir / f"{base_dir.name}_logreg_results_mean_summary.csv"
    combined.to_csv(combined_out, index=False)
    print(f"Wrote {combined_out}")


def main() -> None:
    for base_dir in BASE_DIRS:
        summarize_base_dir(base_dir)


if __name__ == "__main__":
    main()
