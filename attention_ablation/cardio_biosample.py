
"""cardio.py with one bag per biosample_id instead of per donor (Figure S7).

Folds come from $DATA_DIR/cardio/cardio_biosample_id_5folds.csv, whose samples column holds
biosample ids; the h5ad ($DATA_DIR/cardio/cardio_processed.h5ad, obs: biosample_id, disease),
the 3-class task and the macro metrics are those of cardio.py. Outputs are the cardio.py files,
written to $RESULTS_DIR/R2-Q4/Results/Cardio_Biosample/; the "donor" column of
per_donor_predictions.csv holds the biosample id. Checkpoints go to
./checkpoints_cardio_biosample/disease_classification_SEED<seed>/, overwritten fold by fold.
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
from scipy.stats import wilcoxon
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import cellection


def plot_ablation_results(df_ab, save_dir, ABLATION_FRACS):
    """Curve and delta PDFs, ablation_pivot_*.csv and the Wilcoxon printout, as in covid.py."""

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
        std  = sub.groupby("frac_removed")["AUROC"].std()

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
        std  = sub.groupby("frac_removed")["F1"].std()

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
            index=["seed","fold","frac_removed"],
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
    print("\n=== Wilcoxon Test: AUROC ===")
    for frac in ABLATION_FRACS:
        sub = pivot_auc[pivot_auc["frac_removed"] == frac]
        if len(sub) >= 2:
            _, p = wilcoxon(sub["top"], sub["random"])
            print(f"Frac={frac}: p={p:.3e}")

    print("\n=== Wilcoxon Test: AUPRC ===")
    for frac in ABLATION_FRACS:
        sub = pivot_auprc[pivot_auprc["frac_removed"] == frac]
        if len(sub) >= 2:
            _, p = wilcoxon(sub["top"], sub["random"])
            print(f"Frac={frac}: p={p:.3e}")

    print("\n=== Wilcoxon Test: F1 ===")
    for frac in ABLATION_FRACS:
        sub = pivot_f1[pivot_f1["frac_removed"] == frac]
        if len(sub) >= 2:
            _, p = wilcoxon(sub["top"], sub["random"])
            print(f"Frac={frac}: p={p:.3e}")

    pivot_auc.to_csv(f"{save_dir}/ablation_pivot_AUROC.csv", index=False)
    pivot_auprc.to_csv(f"{save_dir}/ablation_pivot_AUPRC.csv", index=False)
    pivot_f1.to_csv(f"{save_dir}/ablation_pivot_F1.csv", index=False)

    return pivot_auc, pivot_f1

def ablate_cells_and_predict(obj, padded_bag, length, attention, remove_frac=0.1, mode="top"):
    """Drop k = int(length * remove_frac) cells ("top" or "random") from one bag and re-score it.

    Arguments as in covid.py. Returns the [n_classes] probability vector, or None when k < 1 or
    no cell is left.
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

        logits = output[0]                 # shape [1, n_classes]
        probs = nn.Softmax(dim=1)(logits)[0].cpu().numpy()  # shape [n_classes]

    return probs


def compute_macro_auprc(y_true_oh, y_pred_proba):
    """Mean over classes of the trapezoidal PR-curve area, from [N, C] one-hot labels and probabilities."""
    n_classes = y_true_oh.shape[1]
    scores = []
    for c in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, c], y_pred_proba[:, c])
        scores.append(auc(recall, precision))
    return np.mean(scores)


def run_ablation(obj, true_df, pred_df, attn_df, fold, seed, sample_key="donor_id", n_classes=3):
    """Top vs random ablation of one fold's test bags; rows go to the global ablation_results."""
    global ablation_results

    y_true_full = np.argmax(true_df.values, axis=1)   # [N]
    y_true_full_oh = np.eye(n_classes)[y_true_full]   # [N, C]
    y_prob_full = pred_df.values                      # [N, C]

    baseline_auroc = roc_auc_score(
        y_true_full_oh,
        y_prob_full,
        average="macro",
        multi_class="ovr",
    )
    baseline_auprc = compute_macro_auprc(y_true_full_oh, y_prob_full)

    # The random control is a single draw per bag and fraction; its spread comes from the
    # repeats over seeds x folds, not from resampling within a fold.
    for frac in ABLATION_FRACS:
        degraded_probs_top = []
        degraded_probs_rand = []
        y_true_used = []

        # i indexes both pred_df rows and obj.mil_dataset bags: obj.inference() walks
        # mil_dataset in order (shuffle=False), so row i and bag i are the same biosample.
        # sample_key is passed in as "biosample_id", so `donor` is a biosample id here.
        for i, donor in enumerate(pred_df.index):
            donor_cells = obj.adata.obs[obj.adata.obs[sample_key] == donor].index
            donor_attn = attn_df.loc[donor_cells]["attention_score_raw"].values

            padded_bag_np = obj.mil_dataset.Xs[i]  # [length, n_features]
            # Xs[i] holds only this bag's `length` cells; despite the name, nothing is padded.
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

        y_true_used = np.array(y_true_used)                    # [M]
        y_true_oh = np.eye(n_classes)[y_true_used]             # [M, C]

        y_top = np.vstack(degraded_probs_top)    # [M, C]
        y_rand = np.vstack(degraded_probs_rand)

        new_auroc_top = roc_auc_score(
            y_true_oh,
            y_top,
            average="macro",
            multi_class="ovr",
        )
        new_auroc_rand = roc_auc_score(
            y_true_oh,
            y_rand,
            average="macro",
            multi_class="ovr",
        )

        new_auprc_top = compute_macro_auprc(y_true_oh, y_top)
        new_auprc_rand = compute_macro_auprc(y_true_oh, y_rand)

        y_pred_top = np.argmax(y_top, axis=1)
        y_pred_rand = np.argmax(y_rand, axis=1)
        new_f1_top = f1_score(y_true_used, y_pred_top, average="macro")
        new_f1_rand = f1_score(y_true_used, y_pred_rand, average="macro")

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


print(torch.cuda.is_available())

adata = sc.read_h5ad(
    f"{DATA_DIR}/cardio/cardio_processed.h5ad"
)
folds = pd.read_csv(
    f"{DATA_DIR}/cardio/cardio_biosample_id_5folds.csv",
    index_col=0,
)

sample_key = "biosample_id"
task_key = "disease"
num_folds = folds["fold"].nunique()
save_dir = (
    f"{RESULTS_DIR}/R2-Q4/Results/Cardio_Biosample"
)
os.makedirs(save_dir, exist_ok=True)

print("Disease counts:", adata.obs[task_key].value_counts())

task_name = 'cardio_cls_biosample'
sample_key = 'biosample_id'
task_key = 'disease'
class_num = 3

results = []

ABLATION = True
# Share of each test bag's cells to remove, k = int(n_cells * frac).
ABLATION_FRACS = [0.05, 0.10, 0.20, 0.25, 0.50]
all_predictions = []  # one frame of unablated per-biosample predictions per fold
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
            save_path="./checkpoints_cardio_biosample",  # own folder: cardio.py uses the same task_key and seeds
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

        classes = list(true_df.columns)
        n_classes = len(classes)

        y_true_oh = true_df.values                 # [N, C]
        y_pred_proba = pred_df.values              # [N, C]
        y_true = np.argmax(y_true_oh, axis=1)      # [N]

        if ABLATION:
            run_ablation(
                obj,
                true_df,
                pred_df,
                attn_df,
                fold,
                SEED,
                sample_key=sample_key,
                n_classes=n_classes,
            )

        all_predictions.append(pd.DataFrame({
            "seed": SEED,
            "fold": fold,
            "donor": pred_df.index,
            "true": y_true,
            **{f"prob_{c}": y_pred_proba[:, i] for i, c in enumerate(classes)}
        }))

        auroc_macro = roc_auc_score(
            y_true_oh,
            y_pred_proba,
            average="macro",
            multi_class="ovr",
        )
        macro_f1 = f1_score(y_true, np.argmax(y_pred_proba, axis=1), average="macro")
        macro_auprc = compute_macro_auprc(y_true_oh, y_pred_proba)

        m = {
            "seed": SEED,
            "fold": fold,
            "AUROC": auroc_macro,
            "MacroF1": macro_f1,
            "MacroAUPRC": macro_auprc,
        }
        results.append(m)


df_ab = pd.DataFrame(ablation_results)
df_ab.to_csv(f"{save_dir}/ablation_results.csv", index=False)
pivot_auc, pivot_f1 = plot_ablation_results(df_ab, save_dir, ABLATION_FRACS)
df_res = pd.DataFrame(results)
df_res[["seed","fold","MacroAUPRC"]].to_csv(f"{save_dir}/MacroAUPRC_values.csv", index=False)
print("Mean MacroAUPRC:", df_res["MacroAUPRC"].mean())


df_pred = pd.concat(all_predictions, ignore_index=True)
df_pred.to_csv(f"{save_dir}/per_donor_predictions.csv", index=False)
