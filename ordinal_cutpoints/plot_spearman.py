"""Spearman correlation per run, learned vs fixed cut points, as a box plot with the runs on top.

Reads haniffa_cellection_metrics_cutpoints_True.csv and
haniffa_cellection_metrics_cutpoints_False.csv from the working directory (one row per run, with
"train_cutpoints" and "spearman" columns) and writes cutpoint_ablation_spearman.pdf there.
learned_cutpoints.py and fixed_cutpoints.py write one file per setting and call the column
"spearman_logits", so these two inputs have to be put together from their output first.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df_true  = pd.read_csv("haniffa_cellection_metrics_cutpoints_True.csv")
df_false = pd.read_csv("haniffa_cellection_metrics_cutpoints_False.csv")

df = pd.concat([df_true, df_false], ignore_index=True)

# Spearman rather than Pearson, since the labels are ranks.
plt.figure(figsize=(5,4))
sns.boxplot(
    data=df,
    x="train_cutpoints",
    y="spearman",
    color="lightgray"
)
sns.stripplot(
    data=df,
    x="train_cutpoints",
    y="spearman",
    color="black",
    jitter=0.15,
    size=6
)

plt.xlabel("Train cutpoints")
plt.ylabel("Spearman correlation")
plt.title("Ordinal performance with vs without trainable cutpoints")
plt.tight_layout()
plt.savefig("cutpoint_ablation_spearman.pdf", dpi=300)
plt.close()
