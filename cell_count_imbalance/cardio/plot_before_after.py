"""Cardio, one bag per donor: grouped bars of per-bin AUROC and MacroF1 before and after downsampling.

Reads 4-bin/per_bin_metrics.csv (cardio/before_downsampling.py) and 4-bin_downsample/per_bin_metrics.csv
(cardio/downsample_and_retrain.py) under $RESULTS_DIR/R1-q8/Results/cardio/CELECTION/, plus
4-bin/donor_bin_map.csv, which only sets the bin order. Each bar is one pooled value, so there are no
error bars. Writes cardio_4bin_auroc_before_after.pdf and cardio_4bin_macrof1_before_after.pdf to
4-bin_plots/.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


base_dir = (
    f"{RESULTS_DIR}/"
    "R1-q8/Results/cardio/CELECTION"
)

path_before = f"{base_dir}/4-bin/per_bin_metrics.csv"
path_after = f"{base_dir}/4-bin_downsample/per_bin_metrics.csv"
donor_bin_map_path = f"{base_dir}/4-bin/donor_bin_map.csv"

print("Before file:", path_before)
print("After file: ", path_after)
print("Bin map:    ", donor_bin_map_path)


df_before = pd.read_csv(path_before)
df_after = pd.read_csv(path_after)

df_before["condition"] = "Before Downsampling"
df_after["condition"] = "After Downsampling"

df_all = pd.concat([df_before, df_after], ignore_index=True)


bin_map = pd.read_csv(donor_bin_map_path, index_col=0).iloc[:, 0]
unique_bins = bin_map.unique()


def left_edge(interval_str):
    """Left edge of an interval label such as "(7209.999, 10181.0]", for numeric bin order."""
    clean = (
        interval_str.replace("(", "")
        .replace("[", "")
        .replace("]", "")
        .replace(")", "")
    )
    left = clean.split(",")[0]
    return float(left)


cat_order = sorted(unique_bins, key=left_edge)
df_all["cell_bin"] = pd.Categorical(
    df_all["cell_bin"],
    categories=cat_order,
    ordered=True,
)


plot_dir = f"{base_dir}/4-bin_plots"
os.makedirs(plot_dir, exist_ok=True)


plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_all,
    x="cell_bin",
    y="AUROC",
    hue="condition",
    palette=["#1f77b4", "#ff7f0e"],
)
plt.title("AUROC")
plt.xlabel("Original Cell Count Bin")
plt.ylabel("AUROC")
plt.xticks(rotation=45, ha="right")
ax.legend(
    title=None,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    ncol=1,
    frameon=True,
    fontsize=8,
    handlelength=1.2,
    labelspacing=0.3,
    borderaxespad=0.0,
)
plt.tight_layout()
plt.savefig(f"{plot_dir}/cardio_4bin_auroc_before_after.pdf", bbox_inches="tight")
plt.show()


plt.figure(figsize=(10, 6))
ax = sns.barplot(
    data=df_all,
    x="cell_bin",
    y="MacroF1",
    hue="condition",
    palette=["#1f77b4", "#ff7f0e"],
)
plt.title("MacroF1")
plt.xlabel("Original Cell Count Bin")
plt.ylabel("Macro-F1")
plt.xticks(rotation=45, ha="right")
ax.legend(
    title=None,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    ncol=1,
    frameon=True,
    fontsize=8,
    handlelength=1.2,
    labelspacing=0.3,
    borderaxespad=0.0,
)
plt.tight_layout()
plt.savefig(f"{plot_dir}/cardio_4bin_macrof1_before_after.pdf", bbox_inches="tight")
plt.show()
