
"""Top-attention vs random cell removal on the Ziegler COVID-19 data (Figure S7a).

If attention marks the cells a prediction relies on, removing a donor's highest-attention cells
should hurt the frozen model more than removing as many random cells. The task here is
disease_id, "case" (COVID-19 or long COVID-19) vs "control", one bag per donor_id. CELLECTION
is trained on the 5 predefined donor folds for seeds 0-9 (50 models), and the best checkpoint
of each fold scores the held-out donors as they are; that is the baseline. Then, at each
fraction in ABLATION_FRACS, every test bag is scored again by the same frozen model with
k = int(n_cells * frac) cells taken out: either the k highest-attention cells ("top") or k
cells drawn at random from the same donor ("random"). Nothing is retrained, and donors with
k < 1 are skipped. AUROC, AUPRC (average precision) and F1 (0.5 threshold on the class-1
probability, i.e. the second one-hot label column) are computed over the fold's test donors.
For each metric and fraction, a paired Wilcoxon signed-rank test of top vs random over the
seed x fold pairs is printed.

Reads $DATA_DIR/covid/covid_Ziegler_processed.h5ad (obs: donor_id, plus disease_id or, if that
column is missing, disease__ontology_label to derive it from) and $DATA_DIR/covid/covid_5folds.csv
(index column, samples = donor id, fold). Writes to $RESULTS_DIR/R2-Q4/Results/Covid/:
  ablation_results.csv                  one row per seed x fold x fraction x setting ("top" /
                                        "random"): AUROC, AUPRC, F1 after removal,
                                        baseline_AUROC, baseline_AUPRC
  ablation_pivot_{AUROC,AUPRC,F1}.csv   top, random and delta = top - random per
                                        seed x fold x fraction
  ablation_{auroc,auprc,f1}_curve.pdf   mean +/- std per fraction, top vs random
  ablation_delta_{auroc,auprc,f1}_barplot.pdf
  MacroAUPRC_values.csv                 seed, fold, MacroAUPRC of the unablated model
  per_donor_predictions.csv             seed, fold, donor, true, prob (unablated model)
No save_path is passed, so checkpoints land in ./disease_id_classification_SEED<seed>/ under the
working directory, and each fold overwrites the one before.
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
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import cellection

# Cutoff on the class-1 probability (second one-hot label column) for the F1 scores.
DECISION_THRESHOLD = 0.5


def plot_ablation_results(df_ab, save_dir, ABLATION_FRACS):
    """Curve and delta bar-plot PDFs plus ablation_pivot_{AUROC,AUPRC,F1}.csv for all runs.

    df_ab holds the ablation_results rows (seed x fold x frac_removed x setting). Also prints
    the Wilcoxon tests. Returns (pivot_auc, pivot_f1).
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    import numpy as np
    import os

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    for setting in ["top", "random"]:
        sub = df_ab[df_ab["setting"] == setting]
        mean = sub.groupby("frac_removed")["AUROC"].mean()
        std = sub.groupby("frac_removed")["AUROC"].std()

        plt.plot(mean.index, mean.values, marker="o", label=setting)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.20)

    plt.xlabel("Fraction of cells removed")
    plt.ylabel("AUROC")
    plt.title("AUROC")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_auroc_curve.pdf")
    plt.close()

    pivot_auc = (
        df_ab.pivot_table(
            index=["seed", "fold", "frac_removed"],
            columns="setting",
            values="AUROC",
        ).reset_index()
    )
    # One row per (seed, fold, frac). delta < 0: losing the top-attention cells cost more than
    # losing as many random cells.
    pivot_auc["delta"] = pivot_auc["top"] - pivot_auc["random"]

    plt.figure(figsize=(6, 5))
    sns.barplot(data=pivot_auc, x="frac_removed", y="delta", color="steelblue")
    plt.axhline(0, color="black")
    plt.ylabel("ΔAUROC (Top − Random)")
    plt.xlabel("Fraction removed")
    plt.title("AUROC")
    plt.ylim(-0.3, 0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_delta_auroc_barplot.pdf")
    plt.close()

    plt.figure(figsize=(6, 5))
    for setting in ["top", "random"]:
        sub = df_ab[df_ab["setting"] == setting]
        mean = sub.groupby("frac_removed")["AUPRC"].mean()
        std = sub.groupby("frac_removed")["AUPRC"].std()

        plt.plot(mean.index, mean.values, marker="o", label=setting)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.20)

    plt.xlabel("Fraction of cells removed")
    plt.ylabel("AUPRC")
    plt.title("AUPRC")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_auprc_curve.pdf")
    plt.close()

    pivot_auprc = (
        df_ab.pivot_table(
            index=["seed", "fold", "frac_removed"],
            columns="setting",
            values="AUPRC",
        ).reset_index()
    )
    pivot_auprc["delta"] = pivot_auprc["top"] - pivot_auprc["random"]

    plt.figure(figsize=(6, 5))
    sns.barplot(data=pivot_auprc, x="frac_removed", y="delta", color="darkorange")
    plt.axhline(0, color="black")
    plt.ylabel("ΔAUPRC (Top − Random)")
    plt.xlabel("Fraction removed")
    plt.title("AUPRC")
    plt.ylim(-0.3, 0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_delta_auprc_barplot.pdf")
    plt.close()

    plt.figure(figsize=(6, 5))
    for setting in ["top", "random"]:
        sub = df_ab[df_ab["setting"] == setting]
        mean = sub.groupby("frac_removed")["F1"].mean()
        std = sub.groupby("frac_removed")["F1"].std()

        plt.plot(mean.index, mean.values, marker="o", label=setting)
        plt.fill_between(mean.index, mean - std, mean + std, alpha=0.20)

    plt.xlabel("Fraction of cells removed")
    plt.ylabel("F1 Score")
    plt.title("MacroF1")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_f1_curve.pdf")
    plt.close()

    pivot_f1 = (
        df_ab.pivot_table(
            index=["seed", "fold", "frac_removed"],
            columns="setting",
            values="F1",
        ).reset_index()
    )
    pivot_f1["delta"] = pivot_f1["top"] - pivot_f1["random"]

    plt.figure(figsize=(6, 5))
    sns.barplot(data=pivot_f1, x="frac_removed", y="delta")
    plt.axhline(0, color="black")
    plt.ylabel("ΔF1 (Top − Random)")
    plt.xlabel("Fraction removed")
    plt.title("MacroF1")
    plt.ylim(-0.3, 0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ablation_delta_f1_barplot.pdf")
    plt.close()

    # Paired over (seed, fold) at each fraction; printed only, not saved.
    print("\n=== Wilcoxon Test Results (AUROC: Top vs Random) ===")
    for frac in ABLATION_FRACS:
        sub = pivot_auc[pivot_auc["frac_removed"] == frac]
        if len(sub) < 2:
            continue
        _, p = wilcoxon(sub["top"], sub["random"])
        print(f"Frac={frac}: p={p:.3e}")

    print("\n=== Wilcoxon Test Results (AUPRC: Top vs Random) ===")
    for frac in ABLATION_FRACS:
        sub = pivot_auprc[pivot_auprc["frac_removed"] == frac]
        if len(sub) < 2:
            continue
        _, p = wilcoxon(sub["top"], sub["random"])
        print(f"Frac={frac}: p={p:.3e}")

    print("\n=== Wilcoxon Test Results (F1: Top vs Random) ===")
    for frac in ABLATION_FRACS:
        sub = pivot_f1[pivot_f1["frac_removed"] == frac]
        if len(sub) < 2:
            continue
        _, p = wilcoxon(sub["top"], sub["random"])
        print(f"Frac={frac}: p={p:.3e}")

    pivot_auc.to_csv(f"{save_dir}/ablation_pivot_AUROC.csv", index=False)
    pivot_auprc.to_csv(f"{save_dir}/ablation_pivot_AUPRC.csv", index=False)
    pivot_f1.to_csv(f"{save_dir}/ablation_pivot_F1.csv", index=False)

    return pivot_auc, pivot_f1


def ablate_cells_and_predict(obj, padded_bag, length, attention, remove_frac=0.1, mode="top"):
    """Drop k = int(length * remove_frac) cells ("top" or "random") from one bag and re-score it.

    padded_bag is the donor's [length, n_features] tensor (unpadded, despite the name) and
    attention its [length] raw scores in the same cell order. Returns P(class 1) as a float, or
    None when k < 1 or no cell is left.
    """
    k = int(length * remove_frac)
    if k < 1:
        return None

    # argsort is ascending, so the last k positions are the top-k attention cells. The random
    # control draws k cells without replacement from the global NumPy RNG.
    if mode == "top":
        idx_remove = np.argsort(attention)[-k:]
    elif mode == "random":
        idx_remove = np.random.choice(np.arange(length), k, replace=False)
    else:
        raise ValueError("mode must be 'top' or 'random'")

    idx_keep = np.array([i for i in range(length) if i not in set(idx_remove)])
    if len(idx_keep) == 0:
        return None

    # Row j of the bag and attention[j] are the same cell; both follow the cell order in
    # obj.adata.obs (see run_ablation).
    idx_keep_t = torch.tensor(idx_keep, dtype=torch.long)
    new_bag = padded_bag[idx_keep_t, :]   # [length', n_features]

    # Same forward pass as obj.inference(), on the frozen best checkpoints, which inference()
    # left on CPU. The encoder maps (n_kept, n_genes) -> (n_kept, hidden); the attention
    # model takes (1, hidden, n_kept).
    input_ = new_bag.unsqueeze(0).permute(0, 2, 1)  # [1, n_features, length']

    with torch.no_grad():
        if obj.InstanceEncoder:
            enc_out = obj.enc_checkpoint(input_.squeeze(0).transpose(0, 1), None)
            enc_out = enc_out.transpose(0, 1).unsqueeze(0)
            output = obj.pn_checkpoint(enc_out)
        else:
            output = obj.pn_checkpoint(input_)

        # output[0] = logits [1, 2]
        logits = output[0]
        prob = nn.Softmax(dim=1)(logits)[0, 1].item()

    return prob


def run_ablation(obj, true_df, pred_df, attn_df, fold, seed, sample_key="donor_id"):
    """Top vs random ablation of one fold's test donors at every fraction in ABLATION_FRACS.

    Appends one "top" and one "random" row per fraction to the global ablation_results, each
    carrying the fold's unablated baseline AUROC and AUPRC.
    """
    global ablation_results

    y_true_full = np.argmax(true_df.values, axis=1)
    y_prob_full = pred_df.values[:, 1]
    baseline_auroc = roc_auc_score(y_true_full, y_prob_full)
    baseline_auprc = average_precision_score(y_true_full, y_prob_full)

    # The random control is a single draw per donor and fraction; its spread comes from the
    # repeats over seeds x folds, not from resampling within a fold.
    for frac in ABLATION_FRACS:
        degraded_probs_top = []
        degraded_probs_rand = []
        y_true_used = []

        # i indexes both pred_df rows and obj.mil_dataset bags: obj.inference() walks
        # mil_dataset in order (shuffle=False), so row i and bag i are the same donor.
        for i, donor in enumerate(pred_df.index):
            donor_cells = obj.adata.obs[obj.adata.obs[sample_key] == donor].index
            donor_attn = attn_df.loc[donor_cells]["attention_score_raw"].values

            padded_bag_np = obj.mil_dataset.Xs[i]  # [length, n_features]
            # Xs[i] holds only this donor's `length` cells; despite the name, nothing is padded.
            length = obj.mil_dataset.ins[i]

            padded_bag = torch.tensor(padded_bag_np, dtype=torch.float32)

            p_top = ablate_cells_and_predict(
                obj, padded_bag, length, donor_attn,
                remove_frac=frac, mode="top"
            )
            p_rand = ablate_cells_and_predict(
                obj, padded_bag, length, donor_attn,
                remove_frac=frac, mode="random"
            )

            if (p_top is None) or (p_rand is None):
                continue

            degraded_probs_top.append(p_top)
            degraded_probs_rand.append(p_rand)
            y_true_used.append(y_true_full[i])

        if len(y_true_used) < 2:
            continue

        y_true_used = np.array(y_true_used)
        y_top = np.array(degraded_probs_top)
        y_rand = np.array(degraded_probs_rand)

        new_auroc_top = roc_auc_score(y_true_used, y_top)
        new_auprc_top = average_precision_score(y_true_used, y_top)
        new_auroc_rand = roc_auc_score(y_true_used, y_rand)
        new_auprc_rand = average_precision_score(y_true_used, y_rand)
        new_f1_top = f1_score(y_true_used, (y_top >= DECISION_THRESHOLD).astype(int))
        new_f1_rand = f1_score(y_true_used, (y_rand >= DECISION_THRESHOLD).astype(int))
        delta_top = baseline_auroc - new_auroc_top
        delta_rand = baseline_auroc - new_auroc_rand

        print(f"[Fold {fold}, Seed {seed}] Remove {int(frac*100)}% cells | "
              f"ΔAUROC Top={delta_top:.4f} | ΔAUROC Random={delta_rand:.4f}")

        ablation_results.append({
        "seed": seed,
        "fold": fold,
        "frac_removed": frac,
        "setting": "top",
        "AUROC": new_auroc_top,
        "AUPRC": new_auprc_top,
        "F1": new_f1_top,
        "baseline_AUROC": baseline_auroc,
        "baseline_AUPRC": baseline_auprc,
        })

        ablation_results.append({
            "seed": seed,
            "fold": fold,
            "frac_removed": frac,
            "setting": "random",
            "AUROC": new_auroc_rand,
            "AUPRC": new_auprc_rand,
            "F1": new_f1_rand,
            "baseline_AUROC": baseline_auroc,
            "baseline_AUPRC": baseline_auprc,
        })

def compute_metrics_binary(y_true, y_prob):
    """Unablated metrics for one fold: AUROC, AUPRC (average precision), MacroAUPRC (mean of the
    two classes' trapezoidal PR-curve areas), MacroF1, and F1 with pos_label 1 (F1_case) and 0
    (F1_control)."""
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    out = {}

    out["AUROC"] = roc_auc_score(y_true, y_prob)
    out["AUPRC"] = average_precision_score(y_true, y_prob)

    precision_pos, recall_pos, _ = precision_recall_curve(y_true, y_prob)
    auprc_pos = auc(recall_pos, precision_pos)

    precision_neg, recall_neg, _ = precision_recall_curve(1 - y_true, 1 - y_prob)
    auprc_neg = auc(recall_neg, precision_neg)

    out["MacroAUPRC"] = (auprc_pos + auprc_neg) / 2

    out["MacroF1"] = f1_score(y_true, y_pred, average="macro")

    out["F1_case"] = f1_score(y_true, y_pred, pos_label=1)
    out["F1_control"] = f1_score(y_true, y_pred, pos_label=0)

    return out


print(torch.cuda.is_available())

adata = sc.read_h5ad(
    f"{DATA_DIR}/covid/covid_Ziegler_processed.h5ad"
)
folds = pd.read_csv(
    f"{DATA_DIR}/covid/covid_5folds.csv",
    index_col=0
)

save_dir = f"{RESULTS_DIR}/R2-Q4/Results/Covid/"
os.makedirs(save_dir, exist_ok=True)

if "disease_id" not in adata.obs.columns:
    adata.obs["disease_id"] = "control"
    adata.obs.loc[
        adata.obs["disease__ontology_label"].isin(["COVID-19","long COVID-19"]),
        "disease_id"
    ] = "case"

sample_key = "donor_id"
task_key = "disease_id"
num_folds = folds["fold"].nunique()
results = []

ABLATION = True
# Share of each test donor's cells to remove, k = int(n_cells * frac).
ABLATION_FRACS = [0.05, 0.10, 0.20, 0.25, 0.50]
all_predictions = []  # one frame of unablated per-donor predictions per fold
ablation_results = []
# NumPy is seeded once per SEED, not per fold; the random-removal control draws from that
# same stream.
for SEED in range(10):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\n=== SEED {SEED} ===")
    for fold in range(1, num_folds+1):

        print(f"\n--- Fold {fold} ---")

        train_donors = folds.loc[folds["fold"] != fold, "samples"]
        test_donors  = folds.loc[folds["fold"] == fold, "samples"]

        adata_train = adata[adata.obs[sample_key].isin(train_donors)]
        adata_test  = adata[adata.obs[sample_key].isin(test_donors)].copy()

        if adata_train.obs[sample_key].nunique() <= 2:
            continue

        obj = cellection.cellectiion_object(
            adata=adata_train,
            task_type="classification",
            task_key=task_key,
            sample_key=sample_key,
            batch_key=None,
            input_type="measurement",
            sparse_input=True,
            InstanceEncoder=True,
            use_tnet=True,
            val_size=0.2,
            aggregator="gated_attention",
            global_features=256,
            attention_dim=64,
            max_epochs=200,
            learning_rate=1e-4,
            batch_size=15,
            seed=SEED,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            hidden_layer=[256,64],
            activation=nn.ReLU(),
            layernorm=True,
            dropout_rate=0.1,
            patience=20,
            save_model=True,
        )

        obj.prepare()
        # Drop training batches with a single bag (b[2] holds the bag sizes). Listing the loader
        # also freezes one shuffle, so every epoch sees the same batches in the same order.
        obj.train_loader = [b for b in obj.train_loader if len(b[2]) > 1]
        obj.train()
        # Not used below: this cell-level split (adata_train_split, adata_val) feeds nothing.
        # Early stopping relies on cellection's own bag-level split (val_size=0.2).
        train_idx, val_idx = train_test_split(
            np.arange(adata_train.n_obs),
            test_size=0.2,
            random_state=SEED,
            stratify=adata_train.obs[task_key]
        )

        adata_train_split = adata_train[train_idx].copy()
        adata_val = adata_train[val_idx].copy()

        obj.adata = adata_test
        obj.prepare_inference()
        sample_meta_all, true_df, pred_df, emb_df, gf_df, attn_df = obj.inference()

        y_true = np.argmax(true_df.values, axis=1)
        y_prob = pred_df.values[:,1]

        if ABLATION:
            run_ablation(obj, true_df, pred_df, attn_df, fold, SEED)
        all_predictions.append(pd.DataFrame({
            "seed":SEED, "fold":fold, "donor":pred_df.index,
            "true":y_true, "prob":y_prob
        }))

        m = compute_metrics_binary(y_true, y_prob)
        m["seed"] = SEED
        m["fold"] = fold
        results.append(m)


df_ab = pd.DataFrame(ablation_results)
df_ab.to_csv(f"{save_dir}/ablation_results.csv", index=False)
pivot_auc, pivot_f1 = plot_ablation_results(df_ab, save_dir, ABLATION_FRACS)
df_res = pd.DataFrame(results)
df_res[["seed","fold","MacroAUPRC"]].to_csv(f"{save_dir}/MacroAUPRC_values.csv", index=False)
print("Mean MacroAUPRC:", df_res["MacroAUPRC"].mean())


df_pred = pd.concat(all_predictions, ignore_index=True)
df_pred.to_csv(f"{save_dir}/per_donor_predictions.csv", index=False)
