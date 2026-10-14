"""sciLaMA arm, seed 18, one timed epoch: the counterpart of timing_raw_expression_seed18_1ep.py.

A copy of train_scilama.py with SEEDS = [18], max_epochs = 1 and the same perf_counter timer
(CUDA-synchronized) around obj.train(); loading the data and the sciMultiLaMA embedding, the donor
split and model setup are outside the timer. It prints
    TIMING_ONLY setting=sciLaMA seed=18 epochs=1 train_seconds=<s> seconds_per_epoch=<s>
and exits before the test-donor scoring. Array task 1 of run_timing_probe.sbatch.

Reads $DATA_DIR/sea-ad/MTG_glia_intersection_norm.h5ad and
$DATA_DIR/sea-ad/MTG_sciMultiLaMA_CELL_embedding.csv. Writes checkpoints to
$RESULTS_DIR/R1-q3/Results/timing_probe/sciLaMA_seed18_1ep/seed_18/ and, if missing,
$RESULTS_DIR/R1-q3/Results/seaad_splits/disease_seed_18_split.csv. Needs `cellection`; uses a GPU
when available.
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import time

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, precision_recall_curve, auc
)
from sklearn.model_selection import train_test_split

import cellection


def compute_macro_auprc(y_true_oh, y_pred_proba):
    """Mean over classes of the one-vs-rest precision-recall AUC (trapezoidal `auc`)."""
    n_classes = y_true_oh.shape[1]
    scores = []
    for c in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, c], y_pred_proba[:, c])
        scores.append(auc(recall, precision))
    return np.mean(scores)


def compute_metrics_binary(y_true, y_prob):
    """Metrics from 0/1 labels and the class-1 score; F1 and confusion matrix use y_prob >= 0.5."""
    y_pred = (y_prob >= 0.5).astype(int)

    out = {}
    out["AUROC"] = roc_auc_score(y_true, y_prob)
    out["AUPRC"] = average_precision_score(y_true, y_prob)

    y_true_oh = np.eye(2)[y_true]
    y_pred_proba = np.column_stack([1 - y_prob, y_prob])
    out["MacroAUPRC"] = compute_macro_auprc(y_true_oh, y_pred_proba)

    out["MacroF1"] = f1_score(y_true, y_pred, average="macro")
    out["F1_case"] = f1_score(y_true, y_pred, pos_label=1)
    out["F1_control"] = f1_score(y_true, y_pred, pos_label=0)
    out["conf_mat"] = confusion_matrix(y_true, y_pred)

    return out


adata = sc.read_h5ad(
    f"{DATA_DIR}/sea-ad/MTG_glia_intersection_norm.h5ad"
)

sci = pd.read_csv(
    f"{DATA_DIR}/sea-ad/MTG_sciMultiLaMA_CELL_embedding.csv",
    index_col=0
)

sci = sci.loc[adata.obs_names]  # same cell order as adata
adata.obsm["embedding"] = sci.values

print("Loaded sciLaMA embeddings:", adata.obsm["embedding"].shape)


sample_key = "donor_id"
task_key = "disease"
save_dir = f"{RESULTS_DIR}/R1-q3/Results/timing_probe/sciLaMA_seed18_1ep"
os.makedirs(save_dir, exist_ok=True)
split_dir = f"{RESULTS_DIR}/R1-q3/Results/seaad_splits"
os.makedirs(split_dir, exist_ok=True)
SEEDS = [18]
results = []
all_predictions = []


for SEED in SEEDS:
    print("\n==============================")
    print(f"        SEED {SEED}")
    print("==============================")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    donor_df = (
        adata.obs[[sample_key, task_key]]
        .drop_duplicates()
        .set_index(sample_key)
    )

    donors = donor_df.index.values
    donor_labels = donor_df[task_key].values

    split_path = os.path.join(split_dir, f"{task_key}_seed_{SEED}_split.csv")
    if os.path.exists(split_path):
        split_df = pd.read_csv(split_path)
        train_donors = split_df.loc[split_df["split"] == "train", sample_key].values
        test_donors = split_df.loc[split_df["split"] == "test", sample_key].values
    else:
        train_donors, test_donors = train_test_split(
            donors,
            test_size=0.20,
            random_state=SEED,
            stratify=donor_labels,
        )
        split_df = pd.DataFrame(
            {
                sample_key: np.concatenate([train_donors, test_donors]),
                "split": ["train"] * len(train_donors) + ["test"] * len(test_donors),
            }
        )
        split_df.to_csv(split_path, index=False)

    adata_train = adata[adata.obs[sample_key].isin(train_donors)].copy()
    adata_test = adata[adata.obs[sample_key].isin(test_donors)].copy()

    print("Train donors:", len(train_donors))
    print("Test donors:", len(test_donors))
    print("Train cells:", adata_train.n_obs)
    print("Test cells:", adata_test.n_obs)

    obj = cellection.cellectiion_object(
        adata=adata_train,
        task_type="classification",
        task_key=task_key,
        sample_key=sample_key,
        batch_key=None,

        input_type="embedding",
        InstanceEncoder=True,
        use_tnet=True,

        val_size=0.2,
        aggregator="gated_attention",
        global_features=256,
        attention_dim=64,
        max_epochs=1,
        learning_rate=1e-4,
        batch_size=15,
        seed=SEED,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        hidden_layer=[256, 32],
        activation=nn.ReLU(),
        layernorm=True,
        dropout_rate=0.1,
        patience=20,

        save_model=True,
        save_path=f"{save_dir}/seed_{SEED}"
    )

    obj.prepare()
    # Timed section: obj.train() for one epoch, with GPU work synchronized on both sides.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    __train_t0 = time.perf_counter()
    obj.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    __train_sec = time.perf_counter() - __train_t0
    print(f"TIMING_ONLY setting=sciLaMA seed={SEED} epochs=1 train_seconds={__train_sec:.3f} seconds_per_epoch={__train_sec / 1:.6f}", flush=True)
    raise SystemExit(0)

    # Test bags are built by hand for embedding input; prepare_inference() / inference() are not
    # used in this arm.
    obj.adata = adata_test

    # Embedding, one-hot label and donor of every test cell
    X = obj.adata.obsm["embedding"]         # shape: (cells, C)
    y = pd.get_dummies(obj.adata.obs[obj.task_key]).values
    meta = obj.adata.obs[obj.sample_key].values

    # One bag per donor
    bags = {}
    labels = {}
    for i in range(len(meta)):
        donor = meta[i]
        if donor not in bags:
            bags[donor] = []
            labels[donor] = y[i]
        bags[donor].append(X[i])

    bag_list = []
    label_list = []
    ins_list = []
    meta_list = []

    for donor in bags:
        arr = np.vstack(bags[donor])      # (N_cells, C)
        bag_list.append(torch.tensor(arr, dtype=torch.float32))
        label_list.append(torch.tensor(labels[donor], dtype=torch.float32))
        ins_list.append(arr.shape[0])
        meta_list.append(donor)

    # Collate fn: zero-pad the bags to (B, N, C). With embedding input and the instance encoder,
    # CELLECTION takes bags as (B, N, C), not (B, C, N).
    def test_collate_fn(batch):
        """Zero-pad bags to the longest bag; returns (padded_bags, labels, lengths, donor ids)."""
        padded_bags = []
        labels = []
        lengths = []
        metas = []
        max_len = max([x[0].shape[0] for x in batch])

        for bag, label, length, donor_id in batch:
            pad = torch.zeros((max_len, bag.shape[1]))
            pad[:bag.shape[0]] = bag
            padded_bags.append(pad)
            labels.append(label)
            lengths.append(length)
            metas.append(donor_id)

        padded_bags = torch.stack(padded_bags)  # (B, N, C)
        labels = torch.stack(labels)
        return padded_bags, labels, lengths, metas

    test_dataset = list(zip(bag_list, label_list, ins_list, meta_list))
    obj.test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_collate_fn
    )
    true_list = []
    pred_list = []
    index_list = []

    obj.pn_checkpoint.eval()
    if obj.InstanceEncoder:
        obj.enc_checkpoint.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for bags, labels, lengths, meta in obj.test_loader:
        length = lengths[0]
        input_all = bags[0, :length, :].to(device)

        with torch.no_grad():
            if obj.InstanceEncoder:
                enc_all = obj.enc_checkpoint(input_all, None)
                out = obj.pn_checkpoint(enc_all.T.unsqueeze(0))
            else:
                input_all_bn = input_all.unsqueeze(0).permute(0, 2, 1)
                out = obj.pn_checkpoint(input_all_bn)

            # The head returns a tuple whose first element is the class output;
            # a plain tensor or a dict with "logits" is accepted as well.
            if isinstance(out, torch.Tensor):
                logits = out

            elif isinstance(out, dict):
                logits = out["logits"]

            elif isinstance(out, tuple):
                logits = out[0]

            else:
                raise ValueError("Unknown output format from model.")

        true_list.append(labels.cpu().numpy())
        pred_list.append(logits.cpu().numpy())
        index_list.extend(meta)

    true_df = pd.DataFrame(np.concatenate(true_list), index=index_list)
    pred_df = pd.DataFrame(np.concatenate(pred_list), index=index_list)

    # y_true: index of the true disease column; y_prob: column 1 of the head output as returned
    # (no softmax applied).
    y_true = np.argmax(true_df.values, axis=1)
    y_prob = pred_df.values[:, 1]

    m = compute_metrics_binary(y_true, y_prob)
    m["seed"] = SEED
    results.append(m)

    all_predictions.append(pd.DataFrame({
        "seed": SEED,
        "donor": pred_df.index,
        "true": y_true,
        "prob": y_prob
    }))


df = pd.DataFrame(results)
df.to_csv(f"{save_dir}/metrics_5seed.csv", index=False)

df_pred = pd.concat(all_predictions, ignore_index=True)
df_pred.to_csv(f"{save_dir}/predictions_5seed.csv", index=False)

print("\n===== FINAL 5-SEED RESULTS =====")
print(df[["seed", "AUROC", "AUPRC", "MacroAUPRC", "MacroF1"]].mean())


