"""COVID cell-count bins, before vs after downsampling: grouped bars of per-bin AUROC and MacroF1.

The two per-bin tables are bin_8/cellection_global_bins.csv (covid/before_downsampling.py) and
bin_8_subsample_retrain/cellection_global_bins_retrain_downsampled.csv (covid/downsample_and_retrain.py)
under $RESULTS_DIR/R1-q8/Results/covid_Ziegler/CELECTION/; bin_8/donor_bin_map.csv is read only to
order the bins. Each bar is a single pooled value, so there are no error bars. Writes
auroc_grouped_bins_6.pdf and macrof1_grouped_bins_6.pdf to the same folder; both show all 8 bins, the
"_6" is just part of the file name.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


path_before = (
    f"{RESULTS_DIR}/R1-q8/Results/covid_Ziegler/CELECTION/bin_8/cellection_global_bins.csv"
)

path_after = (
    f"{RESULTS_DIR}/R1-q8/Results/covid_Ziegler/CELECTION/bin_8_subsample_retrain/cellection_global_bins_retrain_downsampled.csv"
)

donor_bin_map_path = (
    f"{RESULTS_DIR}/"
    "R1-q8/Results/covid_Ziegler/CELECTION/bin_8/"
    "donor_bin_map.csv"
)


df_before = pd.read_csv(path_before)
df_after  = pd.read_csv(path_after)

df_before["condition"] = "Before Downsampling"
df_after["condition"]  = "After Downsampling"

df_all = pd.concat([df_before, df_after], ignore_index=True)


bin_map = pd.read_csv(donor_bin_map_path, index_col=0).iloc[:, 0]
unique_bins = bin_map.unique()

# Bin labels are interval strings such as "(86.0, 120.0]". Sorted as text, "(1070.0, ...]" would come
# before "(120.0, ...]", so they are ordered by their numeric left edge.
def left_edge(interval_str):
    clean = (interval_str.replace("(", "")
                           .replace("[", "")
                           .replace("]", "")
                           .replace(")", ""))
    return float(clean.split(",")[0])

cat_order = sorted(unique_bins, key=left_edge)

df_all["cell_bin"] = pd.Categorical(
    df_all["cell_bin"],
    categories=cat_order,
    ordered=True
)


plot_dir = (
    f"{RESULTS_DIR}/"
    "R1-q8/Results/covid_Ziegler/CELECTION"
)
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
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
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
plt.savefig(f"{plot_dir}/auroc_grouped_bins_6.pdf", bbox_inches="tight")
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
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
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
plt.savefig(f"{plot_dir}/macrof1_grouped_bins_6.pdf", bbox_inches="tight")
plt.show()


print("Saved 6-bin grouped plots:")
print(f"  {plot_dir}/auroc_grouped_bins_6.pdf")
print(f"  {plot_dir}/macrof1_grouped_bins_6.pdf")
