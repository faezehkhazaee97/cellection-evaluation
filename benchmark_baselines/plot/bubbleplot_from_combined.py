"""Figure 2 bubble and mean-rank plots, drawn from all_models_test_summary.csv.

Input: $RESULTS_DIR/R2-q1/Results/all_models_test_summary.csv (plot/combine_all_test_summaries.py).

For each metric (auROC, macroF1, AUPRC = macroAUPRC) the models are ranked within each dataset (1 = best, ties
share the lowest rank) and ordered by mean rank. Two plots per metric: a horizontal bar plot of the mean ranks,
and a model x dataset bubble plot whose area is the value min-max scaled within the dataset and whose colour is
the rank. Values are means over the seed x fold runs; no spread is shown. MMIL rows are drawn if the table has
them (missing Cardio and CardioBiosample values as '-'), but the combined table has none.

Writes {auROC,macroF1,AUPRC}_mean_rankings_barplot.pdf and {auROC,macroF1,AUPRC}_bubble.pdf to
$RESULTS_DIR/R2-q1/Results/plot/.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


INPUT_PATH = Path(
    f"{RESULTS_DIR}/R2-q1/Results/all_models_test_summary.csv"
)
OUTPUT_DIR = Path(f"{RESULTS_DIR}/R2-q1/Results/plot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_RENAME = {
    "Lupus_Disease": "Lupus",
    "LupusRace": "Lupus_Race",
}

DATASET_ORDER = ["Covid", "Lupus", "Lupus_Race", "Cardio", "CardioBiosample"]
DATASET_LABELS = {
    "Covid": "Ziegler\nCOVID-19",
    "Lupus": "Lupus",
    "Lupus_Race": "Ethnicity",
    "Cardio": "Cardio",
    "CardioBiosample": "Cardio\n(biosample-level)",
}

MODEL_LABELS = {
    "CELECTION": "CELLECTION",
    "MMIL": "MMIL",
    "CT_pb": "CT pseudobulk",
    "Pseudobulk": "Pseudobulk",
    "DeepSet": "DeepSet",
    "proto_16": "ProtCell4P (16)",
    "proto_8": "ProtCell4P (8)",
    "CT_proportion": "CT proportion",
    "CloudPred": "CloudPred",
    "proto_4": "ProtCell4P (4)",
}

MODEL_ORDER = [
    "CELECTION",
    "MMIL",
    "CT_pb",
    "Pseudobulk",
    "DeepSet",
    "proto_16",
    "proto_8",
    "CT_proportion",
    "CloudPred",
    "proto_4",
]
MISSING_MARK = "-"
MISSING_MARK_FONTSIZE = 28
MISSING_MARK_MODELS = {"MMIL"}
MISSING_MARK_DATASETS = {"Cardio", "CardioBiosample"}

# Bubble area (points^2): the value min-max scaled to about [0, MAX_BUBBLE_SIZE] within a dataset (BUBBLE_SIZE_EPS
# keeps the lowest value's area above zero); EQUAL_VALUES_BUBBLE_SIZE when all models of a dataset score the same.
MAX_BUBBLE_SIZE = 1000
BUBBLE_SIZE_EPS = 1e-5
EQUAL_VALUES_BUBBLE_SIZE = 300


def load_summary() -> pd.DataFrame:
    """Read the combined summary, coerce the metric columns to numbers and harmonise dataset names."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing combined summary: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    df = df[["model", "dataset", "auROC_test", "macroF1_test", "macroAUPRC_test"]]
    for col in ["auROC_test", "macroF1_test", "macroAUPRC_test"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["dataset"] = df["dataset"].replace(DATASET_RENAME)
    return df


def metric_matrix(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Model x dataset matrix of one metric, in MODEL_ORDER / DATASET_ORDER, without all-NaN models."""
    matrix = df.pivot_table(index="model", columns="dataset", values=metric_col, aggfunc="mean")
    matrix = matrix.reindex(columns=[d for d in DATASET_ORDER if d in matrix.columns])
    matrix = matrix.reindex(index=[m for m in MODEL_ORDER if m in matrix.index])
    matrix = matrix.dropna(axis=0, how="all")
    return matrix


def plot_bubble(metric_df: pd.DataFrame, metric_label: str, cmap: str, out_prefix: str, bar_color: str) -> None:
    """Save the mean-rank bar plot and the bubble plot of one metric matrix (files named <out_prefix>_*.pdf)."""
    rankings = {col: metric_df[col].rank(method="min", ascending=False) for col in metric_df.columns}
    mean_rankings = metric_df.rank(method="min", ascending=False).mean(axis=1)
    sorted_methods = mean_rankings.sort_values(ascending=True).index
    metric_df = metric_df.loc[sorted_methods]
    mean_rankings_sorted = mean_rankings[sorted_methods]

    plt.figure(figsize=(10, 6))
    mean_rankings_sorted.plot(kind="barh", color=bar_color)
    plt.xlabel("Mean Ranking")
    plt.gca().invert_yaxis()
    sns.despine()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{out_prefix}_mean_rankings_barplot.pdf", dpi=300)
    plt.close()

    max_rank = int(max(rank.max() for rank in rankings.values()))
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = None
    for i, method in enumerate(metric_df.index):
        for j, dataset in enumerate(metric_df.columns):
            value = metric_df.loc[method, dataset]
            if pd.isna(value):
                continue
            col_values = metric_df[dataset]
            min_val = col_values.min()
            max_val = col_values.max()
            if max_val == min_val:
                size = EQUAL_VALUES_BUBBLE_SIZE
            else:
                size = MAX_BUBBLE_SIZE * (value - min_val + BUBBLE_SIZE_EPS) / (max_val - min_val)
            rank = rankings[dataset][method]
            if pd.isna(rank):
                continue
            scatter = ax.scatter(
                j,
                i,
                s=size,
                c=[rank],
                cmap=cmap,
                vmin=1,
                vmax=max_rank,
                linewidth=2,
                edgecolor="gray",
            )

    for i, method in enumerate(metric_df.index):
        if method not in MISSING_MARK_MODELS:
            continue
        for j, dataset in enumerate(metric_df.columns):
            if dataset not in MISSING_MARK_DATASETS:
                continue
            if pd.isna(metric_df.loc[method, dataset]):
                ax.text(
                    j,
                    i,
                    MISSING_MARK,
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=MISSING_MARK_FONTSIZE,
                )

    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Ranking")
    ax.set_xticks(range(len(metric_df.columns)))
    ax.set_yticks(range(len(metric_df.index)))
    ax.set_xticklabels([DATASET_LABELS.get(c, c) for c in metric_df.columns], rotation=0, ha="center")
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in metric_df.index])
    ax.set_title(metric_label)
    ax.invert_yaxis()
    sns.despine()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{out_prefix}_bubble.pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    summary_df = load_summary()
    for metric_col, metric_label, cmap, out_prefix, bar_color in [
        ("auROC_test", "auROC", "Reds_r", "auROC", "pink"),
        ("macroF1_test", "macroF1", "Blues_r", "macroF1", "skyblue"),
        ("macroAUPRC_test", "AUPRC", "Greens_r", "AUPRC", "lightgreen"),
    ]:
        metric_df = metric_matrix(summary_df, metric_col)
        plot_bubble(metric_df, metric_label, cmap, out_prefix, bar_color)


if __name__ == "__main__":
    main()
