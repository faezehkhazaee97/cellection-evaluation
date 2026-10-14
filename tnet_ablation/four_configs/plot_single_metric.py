#!/usr/bin/env python3
"""Single-metric companion to ../plot_tnet_bars.py, handy for metrics the Figure S2 panels leave out.

--metric picks any column of metrics_per_fold.csv (default AUROC; AUPRC exists only for the binary
tasks). For each of the five tasks the script reads <results-root>/<Task>/<config>/metrics_per_fold.csv
from the training scripts in this folder and draws one bar per T-Net configuration: mean +/- SD (ddof=1)
across seeds of the per-seed fold mean. A missing file or column gives an empty bar, and a chart is
written for every task even when nothing was found.

--results-root defaults to $RESULTS_DIR/R1-q2/Results/TnetConfigs and --out-dir to its plots/ folder.
Writes <out-dir>/<Task>_<metric>_tnet_bars.pdf and prints how many seeds each configuration had.
"""

# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


CONFIGS = [
    ("tnet_both_off", "Both off"),
    ("tnet_both_on", "Both on"),
    ("tnet1_on_tnet2_off", "TNet1 on, TNet2 off"),
    ("tnet1_off_tnet2_on", "TNet1 off, TNet2 on"),
]

DATASETS = [
    ("Covid", "COVID"),
    ("Cardio", "Cardio"),
    ("CardioBiosample", "Cardio Biosample"),
    ("Lupus_Race", "Lupus Race"),
    ("Lupus_Disease", "Lupus Disease"),
]


def summarize_metric(metrics_path, metric):
    """(mean, SD, n_seeds) over the per-seed fold means.

    Returns None if the file is missing and the string "missing_metric" if the column is.
    """
    if not os.path.exists(metrics_path):
        return None
    df = pd.read_csv(metrics_path)
    if metric not in df.columns:
        return "missing_metric"
    per_seed = df.groupby("seed")[metric].mean()
    mean_val = float(per_seed.mean())
    std_val = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0
    return mean_val, std_val, len(per_seed)


def plot_dataset(dataset_name, dataset_label, results_root, metric, out_dir):
    """Save one task's bar chart and return (pdf path, seeds found per config).

    results_root here is that task's own folder, not the --results-root argument.
    """
    means = []
    stds = []
    labels = []
    legend_labels = []
    seed_counts = []

    for cfg_name, cfg_label in CONFIGS:
        metrics_path = os.path.join(results_root, cfg_name, "metrics_per_fold.csv")
        summary = summarize_metric(metrics_path, metric)
        if summary is None:
            means.append(np.nan)
            stds.append(np.nan)
            labels.append(cfg_name)
            legend_labels.append(cfg_label)
            seed_counts.append(0)
            continue
        if summary == "missing_metric":
            means.append(np.nan)
            stds.append(np.nan)
            labels.append(cfg_name)
            legend_labels.append(cfg_label)
            seed_counts.append(0)
            continue
        mean_val, std_val, seed_count = summary
        means.append(mean_val)
        stds.append(std_val)
        labels.append(cfg_name)
        legend_labels.append(cfg_label)
        seed_counts.append(seed_count)

    x = np.arange(len(labels))
    colors = plt.get_cmap("tab10").colors[: len(labels)]

    plt.figure(figsize=(7, 4))
    plt.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black")
    # Short tick labels in CONFIGS order: "t1/on" is TNet1 only, "t2/on" is TNet2 only.
    plt.xticks(x, ["off/off", "on/on", "t1/on", "t2/on"])
    plt.ylabel(metric)
    plt.title(f"{dataset_label} - {metric} (mean +/- std across seeds)")

    legend_handles = [Patch(facecolor=colors[i], edgecolor="black", label=legend_labels[i])
                      for i in range(len(labels))]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_{metric}_tnet_bars.pdf")
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path, seed_counts


def main():
    parser = argparse.ArgumentParser(
        description="Create per-dataset barplots for TNet config sweeps."
    )
    parser.add_argument(
        "--results-root",
        default=f"{RESULTS_DIR}/R1-q2/Results/TnetConfigs",
        help="Root folder containing dataset subfolders.",
    )
    parser.add_argument(
        "--metric",
        default="AUROC",
        help="Metric column name from metrics_per_fold.csv (e.g., AUROC, AUPRC, MacroAUPRC).",
    )
    parser.add_argument(
        "--out-dir",
        default=f"{RESULTS_DIR}/R1-q2/Results/TnetConfigs/plots",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    for dataset_name, dataset_label in DATASETS:
        results_root = os.path.join(args.results_root, dataset_name)
        out_path, seed_counts = plot_dataset(
            dataset_name, dataset_label, results_root, args.metric, args.out_dir
        )
        print(f"Saved: {out_path} | seeds per config: {seed_counts}")


if __name__ == "__main__":
    main()
