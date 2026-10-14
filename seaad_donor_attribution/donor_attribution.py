"""SEA-AD MTG glia: does a donor's predicted disease probability follow attention or cell-type mix? (Figure S15)

One CELLECTION model is trained (AD vs control on "disease", one bag per donor_id, no seed
loop). Donors are shuffled with np.random.seed(0) and split 80/20 without stratification; the
20% (val_donors / adata_val in the code) are the held-out donors analysed here. cellection holds
out another 20% of the training donors for early stopping (val_size=0.2), and since no model
seed is passed its default seed applies. For each held-out donor the script takes the predicted
probability for class 1 (column 1 of the softmax output; class 1 is the second `disease` category,
`normal`, so P(AD) is one minus it), the donor's attention renormalised to sum to 1
and summed per Supertype, and each Supertype's share of the donor's cells as an abundance
control. Pearson and Spearman correlations of probability with attention fraction and with
population fraction are computed across donors, per Supertype and per major glial class
(Microglia, Oligodendrocyte, OPC, Astrocyte); with fewer than 2 donors they are NaN.

Reads $DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad (obs: donor_id, disease, Supertype;
cells without a Supertype are dropped). Writes to $RESULTS_DIR/R2-q3/Results/:
  donor_predictions.csv                             donor_id, true, pred (threshold 0.5), prob
  attention_aggregated.csv                          donor_id, cell_type (Supertype),
                                                    attention_norm, prob, true
  celltype_attention_and_population_vs_prob.pdf     scatter + regression line per Supertype,
                                                    attention on the top row, population below
  celltype_attention_population_correlations.csv    per Supertype: donor counts, Pearson r and
                                                    Spearman rho for attention and population
  majorcell_attention_and_population_vs_prob.pdf    the same plots per major glial class
  majorcell_attention_population_correlations.csv   the same correlations per major glial class
The checkpoint goes to ./disease_classification_SEED<seed>/ in the working directory, with
<seed> the cellection default.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

import cellection

DATA_PATH = f"{DATA_DIR}/sea-ad/MTG_glia_intersection_norm.h5ad"
SAVE_DIR = f"{RESULTS_DIR}/R2-q3/Results"
os.makedirs(SAVE_DIR, exist_ok=True)

sample_key = "donor_id"
task_key = "disease"        # AD vs control
celltype_key = "Supertype"  # fine glial types

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fraction of donors used for training; the rest are the held-out donors analysed below.
TRAIN_DONOR_FRACTION = 0.8
# Probability cutoff on class 1 (normal) for the predicted label.
DECISION_THRESHOLD = 0.5
# Fewest donors for which a Pearson / Spearman correlation is computed (otherwise NaN).
MIN_DONORS_FOR_CORRELATION = 2

adata = sc.read_h5ad(DATA_PATH)

# Only cells with a Supertype label (the glia) are used.
adata = adata[adata.obs[celltype_key].notna()].copy()

print("Donors:", adata.obs[sample_key].nunique())
print("Cells:", adata.n_obs)

# Despite the val_ names, these 20% of donors are the held-out test set for everything below.
donors = np.array(adata.obs[sample_key].unique())
np.random.seed(0)
np.random.shuffle(donors)

n_train = int(TRAIN_DONOR_FRACTION * len(donors))
train_donors = donors[:n_train]
val_donors   = donors[n_train:]

adata_train = adata[adata.obs[sample_key].isin(train_donors)].copy()
adata_val   = adata[adata.obs[sample_key].isin(val_donors)].copy()

obj = cellection.cellectiion_object(
    adata=adata_train,
    task_type="classification",
    task_key=task_key,
    sample_key=sample_key,
    batch_key=None,
    model_type="classification",
    input_type="measurement",
    sparse_input=True,
    InstanceEncoder=True,
    val_size=0.2,
    aggregator="gated_attention",
    global_features=256,
    attention_dim=64,
    hidden_layer=[256, 32],
    activation=nn.ReLU(),
    dropout_rate=0.1,
    learning_rate=1e-4,
    batch_size=15,
    max_epochs=200,
    patience=20,
    device=DEVICE,
    save_model=True,
)

obj.prepare()
obj.train()

obj.adata = adata_val
obj.prepare_inference()

(
    sample_metadata,
    true_label_df,
    pred_label_df,
    embedding_df,
    global_feature_df,
    attention_mtx_raw_df   # per-cell raw attention, indexed by cell id
) = obj.inference()
attn_df = attention_mtx_raw_df.copy()

attn_df["donor_id"] = adata.obs.loc[attn_df.index, sample_key].values
attn_df["cell_type"] = adata.obs.loc[attn_df.index, celltype_key].values
print(attn_df["cell_type"].value_counts().head(20))

y_true = np.argmax(true_label_df.values, axis=1)
y_prob = pred_label_df.values[:, 1]
y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

pred_df = pd.DataFrame({
    "donor_id": pred_label_df.index,
    "true": y_true,
    "pred": y_pred,
    "prob": y_prob,
})
pred_df.to_csv(f"{SAVE_DIR}/donor_predictions.csv", index=False)

# Rebuilt here; the copy above only fed the cell-type count printout.
attn_df = attention_mtx_raw_df.copy()

ATTN_COL = "attention_score_raw"

attn_df["donor_id"] = adata.obs.loc[attn_df.index, sample_key].values
attn_df["cell_type"] = adata.obs.loc[attn_df.index, celltype_key].values

# Fraction of the donor's total attention carried by each cell.
attn_df["attention_norm"] = (
    attn_df.groupby("donor_id")[ATTN_COL]
           .transform(lambda x: x / x.sum())
)

attn_agg = (
    attn_df.groupby(["donor_id", "cell_type"])["attention_norm"]
           .sum()
           .reset_index()
)

attn_agg = attn_agg.merge(
    pred_df[["donor_id", "prob", "true"]],
    on="donor_id",
    how="left"
)

attn_agg.to_csv(f"{SAVE_DIR}/attention_aggregated.csv", index=False)

# Abundance control on the same held-out donors: Supertype cell count / donor's total cells.
cell_counts = (
    adata_val.obs.groupby([sample_key, celltype_key])
    .size()
    .reset_index(name="cell_count")
)
donor_totals = (
    adata_val.obs.groupby(sample_key)
    .size()
    .reset_index(name="donor_total")
)
cell_prop = cell_counts.merge(donor_totals, on=sample_key, how="left")
cell_prop["cell_prop"] = cell_prop["cell_count"] / cell_prop["donor_total"]
cell_prop = (
    cell_prop
    .rename(columns={sample_key: "donor_id", celltype_key: "cell_type"})
    .merge(pred_df[["donor_id", "prob", "true"]], on="donor_id", how="left")
)

# One column per Supertype: attention fraction vs predicted probability on top, population
# fraction below.
cell_types = sorted(attn_agg["cell_type"].dropna().unique())
n_types = len(cell_types)
if n_types == 0:
    raise ValueError("No cell types found for plotting.")

fig, axes = plt.subplots(
    2,
    n_types,
    figsize=(3.2 * n_types, 6),
    sharex="col"
)
if n_types == 1:
    axes = np.array([[axes[0]], [axes[1]]])

corr_rows = []
for i, ct in enumerate(cell_types):
    attn_ct = (
        attn_agg.query("cell_type == @ct")
        .dropna(subset=["prob", "attention_norm"])
        .sort_values("prob")
    )
    pop_ct = (
        cell_prop.query("cell_type == @ct")
        .dropna(subset=["prob", "cell_prop"])
        .sort_values("prob")
    )

    if len(attn_ct) >= MIN_DONORS_FOR_CORRELATION:
        attn_pr, _ = pearsonr(attn_ct["prob"], attn_ct["attention_norm"])
        attn_sr, _ = spearmanr(attn_ct["prob"], attn_ct["attention_norm"])
    else:
        attn_pr, attn_sr = np.nan, np.nan

    if len(pop_ct) >= MIN_DONORS_FOR_CORRELATION:
        pop_pr, _ = pearsonr(pop_ct["prob"], pop_ct["cell_prop"])
        pop_sr, _ = spearmanr(pop_ct["prob"], pop_ct["cell_prop"])
    else:
        pop_pr, pop_sr = np.nan, np.nan

    corr_rows.append({
        "cell_type": ct,
        "n_donors_attention": len(attn_ct),
        "attention_pearson_r": attn_pr,
        "attention_spearman_r": attn_sr,
        "n_donors_population": len(pop_ct),
        "population_pearson_r": pop_pr,
        "population_spearman_r": pop_sr,
    })

    ax_attn = axes[0, i]
    sns.regplot(
        data=attn_ct,
        x="prob",
        y="attention_norm",
        scatter_kws={"s": 30},
        ax=ax_attn
    )
    ax_attn.set_title(
        f"{ct}\nAttn r={attn_pr:.2f}, ρ={attn_sr:.2f}"
    )
    ax_attn.set_xlabel("")
    ax_attn.set_ylabel("Attention fraction")

    ax_pop = axes[1, i]
    sns.regplot(
        data=pop_ct,
        x="prob",
        y="cell_prop",
        scatter_kws={"s": 30},
        ax=ax_pop
    )
    ax_pop.set_title(
        f"Population r={pop_pr:.2f}, ρ={pop_sr:.2f}"
    )
    ax_pop.set_xlabel("Predicted probability of normal (class 1)")
    ax_pop.set_ylabel("Cell population fraction")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/celltype_attention_and_population_vs_prob.pdf")
plt.close()

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(
    f"{SAVE_DIR}/celltype_attention_population_correlations.csv",
    index=False
)

# Supertype -> major glial class by substring match (e.g. "Micro-PVM_3-SEAAD" -> Microglia,
# "Oligo_2" -> Oligodendrocyte, "OPC_1" -> OPC). Keys are tried in dict order; anything
# unmatched becomes "Other" and is dropped. Fractions are summed over Supertypes per donor.
MAJOR_CELLTYPE_MAP = {
    "Microglia": ["Micro", "PVM"],
    "Oligodendrocyte": ["Oligo"],
    "OPC": ["OPC", "oligodendrocyte precursor"],
    "Astrocyte": ["Astro"],
}

def map_major_celltype(cell_type):
    """First class in MAJOR_CELLTYPE_MAP whose key occurs in the label, else "Other"."""
    cell_str = str(cell_type)
    for major, keys in MAJOR_CELLTYPE_MAP.items():
        if any(k in cell_str for k in keys):
            return major
    return "Other"

attn_major = attn_agg.copy()
attn_major["major_cell_type"] = attn_major["cell_type"].apply(map_major_celltype)
attn_major = attn_major.query("major_cell_type != 'Other'")
attn_major = (
    attn_major.groupby(["donor_id", "major_cell_type"], as_index=False)
    ["attention_norm"]
    .sum()
    .merge(pred_df[["donor_id", "prob", "true"]], on="donor_id", how="left")
)

pop_major = cell_prop.copy()
pop_major["major_cell_type"] = pop_major["cell_type"].apply(map_major_celltype)
pop_major = pop_major.query("major_cell_type != 'Other'")
pop_major = (
    pop_major.groupby(["donor_id", "major_cell_type"], as_index=False)
    ["cell_prop"]
    .sum()
    .merge(pred_df[["donor_id", "prob", "true"]], on="donor_id", how="left")
)

major_types = list(MAJOR_CELLTYPE_MAP.keys())
fig, axes = plt.subplots(
    2,
    len(major_types),
    figsize=(3.2 * len(major_types), 6),
    sharex="col"
)
if len(major_types) == 1:
    axes = np.array([[axes[0]], [axes[1]]])

major_corr_rows = []
for i, mt in enumerate(major_types):
    attn_mt = (
        attn_major.query("major_cell_type == @mt")
        .dropna(subset=["prob", "attention_norm"])
        .sort_values("prob")
    )
    pop_mt = (
        pop_major.query("major_cell_type == @mt")
        .dropna(subset=["prob", "cell_prop"])
        .sort_values("prob")
    )

    if len(attn_mt) >= MIN_DONORS_FOR_CORRELATION:
        attn_pr, _ = pearsonr(attn_mt["prob"], attn_mt["attention_norm"])
        attn_sr, _ = spearmanr(attn_mt["prob"], attn_mt["attention_norm"])
    else:
        attn_pr, attn_sr = np.nan, np.nan

    if len(pop_mt) >= MIN_DONORS_FOR_CORRELATION:
        pop_pr, _ = pearsonr(pop_mt["prob"], pop_mt["cell_prop"])
        pop_sr, _ = spearmanr(pop_mt["prob"], pop_mt["cell_prop"])
    else:
        pop_pr, pop_sr = np.nan, np.nan

    major_corr_rows.append({
        "major_cell_type": mt,
        "n_donors_attention": len(attn_mt),
        "attention_pearson_r": attn_pr,
        "attention_spearman_r": attn_sr,
        "n_donors_population": len(pop_mt),
        "population_pearson_r": pop_pr,
        "population_spearman_r": pop_sr,
    })

    ax_attn = axes[0, i]
    sns.regplot(
        data=attn_mt,
        x="prob",
        y="attention_norm",
        scatter_kws={"s": 30},
        ax=ax_attn
    )
    ax_attn.set_title(
        f"{mt}\nAttn r={attn_pr:.2f}, ρ={attn_sr:.2f}"
    )
    ax_attn.set_xlabel("")
    ax_attn.set_ylabel("Attention fraction")

    ax_pop = axes[1, i]
    sns.regplot(
        data=pop_mt,
        x="prob",
        y="cell_prop",
        scatter_kws={"s": 30},
        ax=ax_pop
    )
    ax_pop.set_title(
        f"Population r={pop_pr:.2f}, ρ={pop_sr:.2f}"
    )
    ax_pop.set_xlabel("Predicted probability of normal (class 1)")
    ax_pop.set_ylabel("Cell population fraction")

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/majorcell_attention_and_population_vs_prob.pdf")
plt.close()

major_corr_df = pd.DataFrame(major_corr_rows)
major_corr_df.to_csv(
    f"{SAVE_DIR}/majorcell_attention_population_correlations.csv",
    index=False
)
