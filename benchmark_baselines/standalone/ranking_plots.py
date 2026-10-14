"""Figure 2 ranking analysis on wide method x dataset metric tables: improvement percentages, tests and plots.

Reads $RESULTS_DIR/R2-q1/Results/Plots/{auROC,macroF1,macroAUPRC}.csv, one row per method (index protocell4p,
protocell8p, protocell16p, ctpb, pb, CT_proportion, CloudPred, DeepSet, CELECTION) and one column per dataset
(Covid, Lupus_Disease, LupusRace, Cardio, Cardio_Biosample). Nothing in this folder writes these tables.

For each metric the script
  * prints how much CELECTION's mean over the five datasets improves (%) on (a) the mean of the other methods,
    (b) the second-best of the other methods by mean and (c) CT_proportion;
  * evaluates Wilcoxon signed-rank and independent t-tests of CELECTION's per-dataset values against (b) and (c),
    as bare expressions, so the results only show in an interactive session;
  * ranks the methods within each dataset (1 = best, ties share the lowest rank) and saves a mean-rank bar plot
    and a bubble plot (area = value min-max scaled within the dataset, colour = rank).
load_and_fix_metric renames datasets and methods to the paper labels for every plot except the macroF1 ones,
which use the table as read.

Writes to $RESULTS_DIR/R2-q1/Code/Plots/Results/:
  {auROC,macroF1,macroAUPRC}_mean_rankings_barplot.png   mean-rank bar plots
  {auROC,macroF1,macroAUPRC}_shorted.png                 bubble plots
"""

# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon, ttest_ind

OUTPUT_DIR = f"{RESULTS_DIR}/R2-q1/Code/Plots/Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bubble area (points^2): the value min-max scaled to about [0, MAX_BUBBLE_SIZE] within a dataset; BUBBLE_SIZE_EPS
# keeps the lowest value's area above zero.
MAX_BUBBLE_SIZE = 1000
BUBBLE_SIZE_EPS = 1e-5


def load_and_fix_metric(path):
    """Read a method x dataset table and rename datasets and methods to the paper labels.

    Keeps only the five dataset columns, in figure order.
    """
    df = pd.read_csv(path, index_col=0)

    df = df.rename(columns={
        "Covid": "Ziegler COVID-19",
        "Lupus_Disease": "Lupus (Disease)",
        "LupusRace": "Lupus (Race)",
        "Cardio": "Cardio",
        "Cardio_Biosample": "Cardio (biosample-level)"
    })

    df = df.rename(index={
        "protocell4p": "ProtCell4P (4)",
        "protocell8p": "ProtCell4P (8)",
        "protocell16p": "ProtCell4P (16)",
        "ctpb": "CT_Pseudobulk",
        "pb": "Pseudobulk",
        "CT_proportion": "CT_proportion",
        "CloudPred": "CloudPred",
        "DeepSet": "DeepSet",
        "CELECTION": "CELECTION"
    })

    desired_order = [
        "Ziegler COVID-19",
        "Lupus (Disease)",
        "Lupus (Race)",
        "Cardio",
        "Cardio (biosample-level)"
    ]
    df = df[desired_order]

    return df


# auROC

metric_df = load_and_fix_metric(
    f"{RESULTS_DIR}/R2-q1/Results/Plots/auROC.csv"
)


metric_df['average'] = metric_df.mean(axis=1)
metric_df_avg_CELECTION = metric_df.loc['CELECTION', 'average']
metric_df_avg_others = metric_df.loc[:, 'average']
metric_df_avg_others = metric_df_avg_others.drop('CELECTION')
metric_df_avg_others = metric_df_avg_others.mean()
# Improvement (%) of CELECTION's mean over the datasets vs the mean of the other methods
improvement_percentage = (metric_df_avg_CELECTION - metric_df_avg_others) / metric_df_avg_others * 100
print(improvement_percentage)


# vs the second-best of the other methods (iloc[1] after sorting them by mean)
metric_df_others = metric_df.drop('CELECTION')
metric_df_others_2nd = metric_df_others.sort_values(by='average', ascending=False).iloc[1]
metric_df_CELECTION = metric_df.loc['CELECTION']
improvement_percentage_2nd = (metric_df_CELECTION['average'] - metric_df_others_2nd['average']) / metric_df_others_2nd['average'] * 100
print(improvement_percentage_2nd)


# vs CT_proportion
metric_df_others_ct_proportion = metric_df_others.loc['CT_proportion']
improvement_percentage_ct_proportion = (metric_df_CELECTION['average'] - metric_df_others_ct_proportion['average']) / metric_df_others_ct_proportion['average'] * 100
print(improvement_percentage_ct_proportion)

# Bare expressions: the test results only show in an interactive session
wilcoxon(metric_df_CELECTION.drop('average'), metric_df_others_ct_proportion.drop('average'))
wilcoxon(metric_df_CELECTION.drop('average'), metric_df_others_2nd.drop('average'))
ttest_ind(metric_df_CELECTION.drop('average'), metric_df_others_ct_proportion.drop('average'))
ttest_ind(metric_df_CELECTION.drop('average'), metric_df_others_2nd.drop('average'))


# auROC: mean-rank bar plot and bubble plot (reloaded without the 'average' column)
metric_df = load_and_fix_metric(
    f"{RESULTS_DIR}/R2-q1/Results/Plots/auROC.csv"
)

# Rank the methods within each dataset (1 = best; ties share the lowest rank)
rankings = {}
for col in metric_df.columns:
    rankings[col] = metric_df[col].rank(method='min', ascending=False).astype(int)

mean_rankings = metric_df.rank(method='min', ascending=False).mean(axis=1)

sorted_methods = mean_rankings.sort_values(ascending=True).index
metric_df = metric_df.loc[sorted_methods]

mean_rankings_sorted = mean_rankings[sorted_methods]

plt.figure(figsize=(10, 6))
mean_rankings_sorted.plot(kind='barh', color='pink')
plt.xlabel('Mean Ranking')
plt.gca().invert_yaxis()
sns.despine()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/auROC_mean_rankings_barplot.png", dpi=300)
plt.show()

# vmax=8 with nine methods: rank 9 is drawn in the rank-8 colour
fig, ax = plt.subplots(figsize=(10, 8))
for i, method in enumerate(metric_df.index):
    for j, metric in enumerate(metric_df.columns):
        value = metric_df.loc[method, metric]
        rank = rankings[metric][method]
        size = MAX_BUBBLE_SIZE * (value - metric_df[metric].min() + BUBBLE_SIZE_EPS) / (metric_df[metric].max() - metric_df[metric].min())
        scatter = ax.scatter(j, i, s=size,
                             c=[rank], cmap='Reds_r',
                             vmin=1, vmax=8,
                             linewidth=2,
                             edgecolor='gray')


cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Ranking')
ax.set_xticks(range(len(metric_df.columns)))
ax.set_yticks(range(len(metric_df.index)))
ax.set_xticklabels(metric_df.columns, rotation=45, ha='right')
ax.set_yticklabels(metric_df.index)
ax.invert_yaxis()
sns.despine()
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/auROC_shorted.png", dpi=300)


# macroF1
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

metric_df = load_and_fix_metric(
    f"{RESULTS_DIR}/R2-q1/Results/Plots/macroF1.csv"
)


metric_df['average'] = metric_df.mean(axis=1)
metric_df_avg_CELECTION = metric_df.loc['CELECTION', 'average']
metric_df_avg_others = metric_df.loc[:, 'average']
metric_df_avg_others = metric_df_avg_others.drop('CELECTION')
metric_df_avg_others = metric_df_avg_others.mean()
# Improvement (%) of CELECTION's mean over the datasets vs the mean of the other methods
improvement_percentage = (metric_df_avg_CELECTION - metric_df_avg_others) / metric_df_avg_others * 100
print(improvement_percentage)


# vs the second-best of the other methods (iloc[1] after sorting them by mean)
metric_df_others = metric_df.drop('CELECTION')
metric_df_others_2nd = metric_df_others.sort_values(by='average', ascending=False).iloc[1]
metric_df_CELECTION = metric_df.loc['CELECTION']
improvement_percentage_2nd = (metric_df_CELECTION['average'] - metric_df_others_2nd['average']) / metric_df_others_2nd['average'] * 100
print(improvement_percentage_2nd)

# vs CT_proportion
metric_df_others_ct_proportion = metric_df_others.loc['CT_proportion']
improvement_percentage_ct_proportion = (metric_df_CELECTION['average'] - metric_df_others_ct_proportion['average']) / metric_df_others_ct_proportion['average'] * 100
print(improvement_percentage_ct_proportion)

# Bare expressions: the test results only show in an interactive session
from scipy.stats import wilcoxon
wilcoxon(metric_df_CELECTION.drop('average'), metric_df_others_ct_proportion.drop('average'))
wilcoxon(metric_df_CELECTION.drop('average'), metric_df_others_2nd.drop('average'))
from scipy.stats import ttest_ind
ttest_ind(metric_df_CELECTION.drop('average'), metric_df_others_ct_proportion.drop('average'))
ttest_ind(metric_df_CELECTION.drop('average'), metric_df_others_2nd.drop('average'))


# macroF1: mean-rank bar plot and bubble plot (table used as read from the CSV, without load_and_fix_metric)
metric_df = pd.read_csv(f'{RESULTS_DIR}/R2-q1/Results/Plots/macroF1.csv', index_col=0)
# Rank the methods within each dataset (1 = best; ties share the lowest rank)
rankings = {}
for col in metric_df.columns:
    rankings[col] = metric_df[col].rank(method='min', ascending=False).astype(int)

mean_rankings = metric_df.rank(method='min', ascending=False).mean(axis=1)

sorted_methods = mean_rankings.sort_values(ascending=True).index
metric_df = metric_df.loc[sorted_methods]

mean_rankings_sorted = mean_rankings[sorted_methods]

plt.figure(figsize=(10, 6))
mean_rankings_sorted.plot(kind='barh', color='skyblue')
plt.xlabel('Mean Ranking')
plt.gca().invert_yaxis()
sns.despine()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/macroF1_mean_rankings_barplot.png", dpi=300)
plt.show()

# vmax=8 with nine methods: rank 9 is drawn in the rank-8 colour
fig, ax = plt.subplots(figsize=(10, 8))
for i, method in enumerate(metric_df.index):
    for j, metric in enumerate(metric_df.columns):
        value = metric_df.loc[method, metric]
        rank = rankings[metric][method]
        size = MAX_BUBBLE_SIZE * (value - metric_df[metric].min() + BUBBLE_SIZE_EPS) / (metric_df[metric].max() - metric_df[metric].min())
        scatter = ax.scatter(j, i, s=size,
                             c=[rank], cmap='Blues_r',
                             vmin=1, vmax=8,
                             linewidth=2,
                             edgecolor='gray')


cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Ranking')
ax.set_xticks(range(len(metric_df.columns)))
ax.set_yticks(range(len(metric_df.index)))
ax.set_xticklabels(metric_df.columns, rotation=45, ha='right')
ax.set_yticklabels(metric_df.index)
ax.invert_yaxis()
sns.despine()
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/macroF1_shorted.png", dpi=300)


# macroAUPRC

metric_df = load_and_fix_metric(
    f"{RESULTS_DIR}/R2-q1/Results/Plots/macroAUPRC.csv"
)

# Improvement (%) of CELECTION's mean over the datasets
metric_df['average'] = metric_df.mean(axis=1)
metric_df_avg_CELECTION = metric_df.loc['CELECTION', 'average']

metric_df_avg_others = metric_df['average'].drop('CELECTION').mean()
print("MacroAUPRC % improvement over overall average:",
      (metric_df_avg_CELECTION - metric_df_avg_others) / metric_df_avg_others * 100)

# vs the second-best of the other methods (iloc[1] after sorting them by mean)
metric_df_others = metric_df.drop('CELECTION')
metric_df_others_2nd = metric_df_others.sort_values(by='average', ascending=False).iloc[1]
metric_df_CELECTION = metric_df.loc['CELECTION']

print("MacroAUPRC % improvement over 2nd best:",
      (metric_df_CELECTION['average'] - metric_df_others_2nd['average']) / metric_df_others_2nd['average'] * 100)

# vs CT_proportion
metric_df_others_ct_proportion = metric_df_others.loc['CT_proportion']
print("MacroAUPRC % improvement over CT_proportion:",
      (metric_df_CELECTION['average'] - metric_df_others_ct_proportion['average']) /
      metric_df_others_ct_proportion['average'] * 100)

# Bare expressions: the test results only show in an interactive session
from scipy.stats import wilcoxon, ttest_ind
wilcoxon(metric_df_CELECTION.drop('average'), metric_df_others_ct_proportion.drop('average'))
wilcoxon(metric_df_CELECTION.drop('average'), metric_df_others_2nd.drop('average'))
ttest_ind(metric_df_CELECTION.drop('average'), metric_df_others_ct_proportion.drop('average'))
ttest_ind(metric_df_CELECTION.drop('average'), metric_df_others_2nd.drop('average'))

# macroAUPRC: mean-rank bar plot (reloaded without the 'average' column)

metric_df = load_and_fix_metric(
    f"{RESULTS_DIR}/R2-q1/Results/Plots/macroAUPRC.csv"
)

rankings = {}
for col in metric_df.columns:
    rankings[col] = metric_df[col].rank(method='min', ascending=False).astype(int)

mean_rankings = metric_df.rank(method='min', ascending=False).mean(axis=1)
sorted_methods = mean_rankings.sort_values(ascending=True).index
metric_df = metric_df.loc[sorted_methods]
mean_rankings_sorted = mean_rankings.loc[sorted_methods]

plt.figure(figsize=(10, 6))
mean_rankings_sorted.plot(kind='barh', color='lightgreen')
plt.xlabel("Mean Ranking")
plt.gca().invert_yaxis()
sns.despine()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/macroAUPRC_mean_rankings_barplot.png", dpi=300)
plt.show()


# macroAUPRC: bubble plot

fig, ax = plt.subplots(figsize=(10, 8))

for i, method in enumerate(metric_df.index):
    for j, metric in enumerate(metric_df.columns):
        value = metric_df.loc[method, metric]
        rank = rankings[metric][method]
        size = MAX_BUBBLE_SIZE * (value - metric_df[metric].min() + BUBBLE_SIZE_EPS) / \
               (metric_df[metric].max() - metric_df[metric].min())

        scatter = ax.scatter(
            j, i, s=size,
            c=[rank], cmap="Greens_r",
            vmin=1, vmax=len(metric_df.index),
            linewidth=2, edgecolor="gray"
        )

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Ranking")

ax.set_xticks(range(len(metric_df.columns)))
ax.set_yticks(range(len(metric_df.index)))
ax.set_xticklabels(metric_df.columns, rotation=45, ha="right")
ax.set_yticklabels(metric_df.index)
ax.invert_yaxis()
sns.despine()
plt.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/macroAUPRC_shorted.png", dpi=300)
plt.show()
