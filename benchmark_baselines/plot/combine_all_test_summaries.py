#!/usr/bin/env python3
"""Merge the seven per-model test summaries into the table behind Figure 2 and Table S3.

Sources, under $RESULTS_DIR/R2-q1/Results/ (SUMMARY_SOURCES; a missing file is reported and skipped):
  ct_pb/ct_pb_logreg_results_mean_summary.csv             these three from
  proportion/proportion_logreg_results_mean_summary.csv   ct_pseudobulk/summarize_linear_baselines.py
  pseudobulk/pseudobulk_logreg_results_mean_summary.csv
  CELECTION/celection_test_summary.csv                    celection/summarize.py
  4_ProtoCell4P/protocell4p_test_summary.csv              protocell4p/summarize.py
  2_DeepSet/deepset_test_summary.csv                      deepset/summarize.py
  cloudpred/cloudpred_test_summary.csv                    cloudpred/summarize.py
The linear baselines are relabelled CT_pb, CT_proportion and Pseudobulk, dataset names are harmonised
(Lupus_Disease -> Lupus, LupusRace -> Lupus_Race), and rows are sorted by MODEL_ORDER, then DATASET_ORDER.

Writes $RESULTS_DIR/R2-q1/Results/all_models_test_summary.csv (model, dataset, auROC_test, macroF1_test,
macroAUPRC_test). MMIL is not part of it; see mmil_baseline/.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path(
    f"{RESULTS_DIR}/R2-q1/Results/all_models_test_summary.csv"
)

SUMMARY_SOURCES = [
    {
        "path": Path(
            f"{RESULTS_DIR}/R2-q1/Results/ct_pb/ct_pb_logreg_results_mean_summary.csv"
        ),
        "model_override": "CT_pb",
    },
    {
        "path": Path(
            f"{RESULTS_DIR}/R2-q1/Results/proportion/proportion_logreg_results_mean_summary.csv"
        ),
        "model_override": "CT_proportion",
    },
    {
        "path": Path(
            f"{RESULTS_DIR}/R2-q1/Results/pseudobulk/pseudobulk_logreg_results_mean_summary.csv"
        ),
        "model_override": "Pseudobulk",
    },
    {
        "path": Path(
            f"{RESULTS_DIR}/R2-q1/Results/CELECTION/celection_test_summary.csv"
        ),
        "model_override": None,
    },
    {
        "path": Path(
            f"{RESULTS_DIR}/R2-q1/Results/4_ProtoCell4P/protocell4p_test_summary.csv"
        ),
        "model_override": None,
    },
    {
        "path": Path(
            f"{RESULTS_DIR}/R2-q1/Results/2_DeepSet/deepset_test_summary.csv"
        ),
        "model_override": None,
    },
    {
        "path": Path(
            f"{RESULTS_DIR}/R2-q1/Results/cloudpred/cloudpred_test_summary.csv"
        ),
        "model_override": None,
    },
]

DATASET_RENAME = {
    "Lupus_Disease": "Lupus",
    "LupusRace": "Lupus_Race",
}

DATASET_ORDER = ["Covid", "Lupus", "Lupus_Race", "Cardio", "CardioBiosample"]
MODEL_ORDER = [
    "CELECTION",
    "CT_pb",
    "Pseudobulk",
    "DeepSet",
    "proto_16",
    "proto_8",
    "CT_proportion",
    "CloudPred",
    "proto_4",
]


def load_summaries() -> pd.DataFrame:
    """Read, relabel, concatenate and sort the available summary files."""
    frames = []
    for source in SUMMARY_SOURCES:
        path = source["path"]
        if not path.exists():
            print(f"Skipping missing summary: {path}")
            continue
        df = pd.read_csv(path)
        df = df[["model", "dataset", "auROC_test", "macroF1_test", "macroAUPRC_test"]]
        if source["model_override"]:
            df["model"] = source["model_override"]
        df["dataset"] = df["dataset"].replace(DATASET_RENAME)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No summary files found to combine.")
    combined = pd.concat(frames, ignore_index=True)
    combined["dataset"] = pd.Categorical(combined["dataset"], DATASET_ORDER, ordered=True)
    combined["model"] = pd.Categorical(combined["model"], MODEL_ORDER, ordered=True)
    combined = combined.sort_values(["model", "dataset"])
    return combined.reset_index(drop=True)


def main() -> None:
    combined = load_summaries()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
