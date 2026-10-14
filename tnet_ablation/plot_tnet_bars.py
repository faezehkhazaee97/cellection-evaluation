"""Figure S2: for each task, AUROC, MacroF1 and MacroAUPRC of the four T-Net configurations side by side.

Reads $RESULTS_DIR/R1-q2/Results/TnetConfigs/<Task>/<config>/metrics_per_fold.csv from the
four_configs/ runs. Each bar is the mean +/- SD (ddof=1) across seeds of the per-seed fold mean, so the
error bar shows seed-to-seed spread rather than fold spread; a file without a seed column falls back to
mean +/- SD over all rows. Tasks with no results folder are skipped, and a missing configuration or
metric leaves an empty bar.

Writes TnetConfigs/plots/<Task>_tnet_metrics.pdf; the plots folder is created on import.
"""

# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


RESULTS_ROOT = f"{RESULTS_DIR}/R1-q2/Results/TnetConfigs"
OUT_DIR = os.path.join(RESULTS_ROOT, "plots")
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    ("Covid", "COVID"),
    ("Cardio", "Cardio"),
    ("CardioBiosample", "Cardio Biosample"),
    ("Lupus_Race", "Lupus Race"),
    ("Lupus_Disease", "Lupus Disease"),
]

# (results subfolder, legend label, x-tick label as TNet1/TNet2 state)
CONFIGS = [
    ("tnet_both_off", "Both off", "off/off"),
    ("tnet_both_on", "Both on", "on/on"),
    ("tnet1_on_tnet2_off", "TNet1 on / TNet2 off", "on/off"),
    ("tnet1_off_tnet2_on", "TNet1 off / TNet2 on", "off/on"),
]

METRICS = ["AUROC", "MacroF1", "MacroAUPRC"]

PLOT_PALETTE = ["#1F77B4", "#FF7F0E", "#2CA02C", "#9467BD"]


def summarize_metric(metrics_path, metric):
    """(mean, SD) of `metric` in one metrics_per_fold.csv, or None if the file or column is missing."""
    if not os.path.isfile(metrics_path):
        return None
    df = pd.read_csv(metrics_path)
    if metric not in df.columns:
        return None
    # Folds are averaged within each seed first, so the SD is across seeds.
    if "seed" in df.columns:
        per_seed = df.groupby("seed")[metric].mean()
        mean_val = float(per_seed.mean())
        std_val = float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0
    else:
        mean_val = float(df[metric].mean())
        std_val = float(df[metric].std())
    return mean_val, std_val


def plot_dataset(dataset_name, dataset_label):
    """Three-panel chart for one task, saved as OUT_DIR/<dataset_name>_tnet_metrics.pdf."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(12, 4.5))
    if len(METRICS) == 1:
        axes = [axes]

    for idx, metric in enumerate(METRICS):
        means = []
        stds = []
        xtick_labels = []
        for cfg_name, _, cfg_short in CONFIGS:
            metrics_path = os.path.join(
                RESULTS_ROOT, dataset_name, cfg_name, "metrics_per_fold.csv"
            )
            summary = summarize_metric(metrics_path, metric)
            if summary is None:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                mean_val, std_val = summary
                means.append(mean_val)
                stds.append(std_val)
            xtick_labels.append(cfg_short)

        ax = axes[idx]
        x = np.arange(len(CONFIGS))
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=PLOT_PALETTE[: len(CONFIGS)],
            edgecolor="black",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=0)
        ax.set_title(metric, pad=10)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    legend_handles = [
        Patch(facecolor=PLOT_PALETTE[i], edgecolor="black", label=CONFIGS[i][1])
        for i in range(len(CONFIGS))
    ]
    fig.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    fig.suptitle(dataset_label, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.85, 0.92])

    out_path = os.path.join(OUT_DIR, f"{dataset_name}_tnet_metrics.pdf")
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    any_plotted = False
    for dataset_name, dataset_label in DATASETS:
        dataset_dir = os.path.join(RESULTS_ROOT, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        plot_dataset(dataset_name, dataset_label)
        any_plotted = True
    if not any_plotted:
        print("No datasets found. Check RESULTS_ROOT path.")
        return
    print(f"Saved plots to: {OUT_DIR}")


if __name__ == "__main__":
    main()
