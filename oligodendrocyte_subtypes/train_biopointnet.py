"""Oligodendrocyte-only disease classifier on SEA-AD MTG, with bioPointNet instead of CELLECTION.

Run after subset_oligodendrocytes.py. Each donor (obs["donor_id"]) is a bag of its
oligodendrocytes, and a cell's features are its full expression vector plus one-hot sex columns.
The bag label is one-hot obs["disease"] (2 classes, pd.get_dummies order). For SEED 0-4: torch
random_split of donors 80/20 into train/validation, CellEncoder (hidden [32]) + PointNetClassHead
with gated attention, Adam lr 1e-3, 15 bags per batch, up to 200 epochs with early stopping on
validation loss (patience 25). Bags with a single cell are skipped everywhere.

After training, AUROC and macro F1 are computed for both splits and the training donors' per-cell
attention is plotted on a sciMultiLaMA UMAP of their cells (all cells, per class, per donor).
Two optional blocks: a Supertype UMAP of all donors (MAKE_ALL_DONOR_UMAP), and the top TOP_FRAC
(10%) attention cells of every donor, grouped by predicted class (MAKE_ALL_DONOR_TOP10_PLOTS;
POS_LABEL must be one of the obs["disease"] categories).

bioPointNet_Apr2025.py has to be importable (put its folder, e.g. $BIOPOINTNET_ROOT, on
PYTHONPATH); its star import also supplies torch, nn, DataLoader and random_split. Uses a GPU when
available.

Reads MTG_oligo_sciMultiLaMA.h5ad (obs donor_id, disease, sex, Supertype; obsm["X_sciMultiLaMA"])
and MTG_oligo_metadata.csv from $RESULTS_DIR/R1-q12/Data/. Writes to
$RESULTS_DIR/R1-q12/Data/MTG/MTG_cls/:
  CLASSIFIER_MTG_<SEED>.pt, ENCODER_MTG_<SEED>.pt  best-validation checkpoints
  RES_STAT_MTG_<SEED>.csv                          auROC / macroF1 for train and validation
  true_/pred_label_tr_MTG_<SEED>.csv               per training donor; pred_label_tr_MTG_<SEED>_.csv
                                                   is the one-hot arg-max prediction
  true_/pred_label_{tr,val}_MTG_<SEED>.png         label heatmaps
  Adata_Tr_MTG_<SEED>.h5ad                         training cells with attention scores in obs
  figures/umap_*.png
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import os
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns

from bioPointNet_Apr2025 import *


data_source = 'MTG'

sample_key = 'donor_id'
task_key = 'disease'
cov_key = 'sex'
ct_key = 'cell_type'
ct_key_fine = 'Supertype'

class_num = 2

path = f'{RESULTS_DIR}/R1-q12/Data/'

output_dir = os.path.join(path, data_source, f"{data_source}_cls")
figures_dir = os.path.join(output_dir, "figures")
os.makedirs(figures_dir, exist_ok=True)
sc.settings.figdir = figures_dir

MAKE_ALL_DONOR_UMAP = True
ALL_DONOR_UMAP_SEED = 0
MAKE_ALL_DONOR_TOP10_PLOTS = True
TOP_FRAC = 0.10
POS_LABEL = "dementia"  # the AD category of SEA-AD obs["disease"] (categories: dementia, normal)

TRAIN_FRACTION = 0.8
# Clip for the z-scored and cell-number-scaled attention shown on the UMAPs.
ATTENTION_CLIP_MAX = 10


adata = sc.read_h5ad(f'{RESULTS_DIR}/R1-q12/Data/MTG_oligo_sciMultiLaMA.h5ad')

# Keep, in this order, the cells listed in the oligodendrocyte metadata table.
metadata_all = pd.read_csv(path + 'MTG_oligo_metadata.csv', index_col=0)
adata = adata[metadata_all.index.tolist()]
metadata = adata.obs.copy()

if MAKE_ALL_DONOR_UMAP:
    if 'X_sciMultiLaMA' not in adata.obsm:
        raise ValueError("X_sciMultiLaMA not found; run subset_oligodendrocytes.py to add embeddings.")
    sc.pp.neighbors(adata, n_neighbors=25, use_rep='X_sciMultiLaMA')
    sc.tl.umap(adata, random_state=ALL_DONOR_UMAP_SEED)
    sc.pl.umap(
        adata,
        color=['Supertype'],
        legend_loc='right margin',
        palette='tab10',
        ncols=1,
        wspace=0.5,
        save='_supertype_MTG_all_donors.png',
    )

Ns = adata.X.copy()
from scipy import sparse

Ns = adata.X.copy()
if sparse.issparse(Ns):
    Ns = Ns.toarray()
Ns_df = pd.DataFrame(Ns, index=metadata.index.tolist())
Ys_df = pd.get_dummies(adata.obs[task_key].copy()).astype(int)
Cs_df = pd.get_dummies(adata.obs[cov_key].copy()).astype(int)
# Instance features: all genes plus one-hot sex columns.
Ns_df = pd.concat([Ns_df, Cs_df], axis=1)
# One bag per donor: Xs[i] is (n_cells_i x n_features), Ys is (n_donors x 2) one-hot,
# ins[i] = n_cells_i, meta_ids[i] = the donor's cell ids (used later to map attention back to cells).
Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata, bag_column=sample_key)
print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")

mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
train_size = int(TRAIN_FRACTION * len(mil_dataset))
val_size = len(mil_dataset) - train_size

for SEED in range(5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def MIL_Collate_fn(batch):
        """Zero-pad bags to the longest in the batch and move them to `device`.

        Returns (padded_bags, labels, lengths, metadata_idx); lengths are the bag sizes.
        """
        bags, labels, ins, metadata_idx= zip(*batch)
        lengths = ins
        max_length = max(ins)
        padded_bags = torch.zeros((len(bags), max_length, bags[0].shape[1]), dtype=torch.float32)
        for i, bag in enumerate(bags):
            padded_bags[i, :len(bag)] = bag
        labels = torch.stack(labels)
        return padded_bags.to(device), labels.to(device), lengths, metadata_idx
    print(SEED)
    torch.manual_seed(SEED)
    set_seed(SEED)
    train_dataset, val_dataset = random_split(mil_dataset, [train_size, val_size])
    print(f"Training set size: {len(train_dataset)}", f"Validation set size: {len(val_dataset)}")
    batch_size = 15
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MIL_Collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False, collate_fn=MIL_Collate_fn)
    set_seed(SEED)
    enc = CellEncoder(input_dim=Ns_df.shape[1], input_batch_num=0, hidden_layer=[32], 
                    layernorm=True, activation=nn.ReLU(), batchnorm=False, dropout_rate=0, 
                    add_linear_layer=True, clip_threshold=None)
    enc.apply(init_weights)
    enc.to(device)
    pn_cls = PointNetClassHead(input_dim=32, k=class_num, global_features=128, attention_dim=32, agg_method="gated_attention")
    pn_cls.apply(init_weights)
    pn_cls.to(device)
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(list(pn_cls.parameters())+list(enc.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    transformation_function = nn.Softmax(dim=1)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    best_model_state = None
    epochs = 200
    # The validation loader has a single batch (batch_size = val_size); it is unpacked once here
    # and reused for the per-epoch validation loss and the final validation metrics.
    for val_padded_bags, val_labels, val_lengths, val_id in val_loader:
            print(len(val_lengths))
    for epoch in range(epochs):
        for padded_bags, labels, lengths, _ in train_loader:
            optimizer.zero_grad()
            loss_tr = 0
            # Bags are processed one at a time (padding removed); bags with a single cell are skipped.
            # input_tr: (1, n_features, n_cells). The encoder takes (n_cells, n_features) and returns
            # (n_cells, 32), which is reshaped to (1, 32, n_cells) for the PointNet head.
            for idx in range(len(lengths)):
                length = lengths[idx]
                if length == 1:
                    continue
                elif length > 1:
                    input_tr = padded_bags[idx, :length,:].unsqueeze(0).permute(0, 2, 1)
                    cls, embedding, global_features, attn_weights, crit_idxs = pn_cls(enc(input_tr.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
                    pred_label = transformation_function(cls)
                    true_label = labels[idx]
                    loss_per_sample = criterion(pred_label, torch.max(true_label.reshape(-1,class_num),1)[1])
                    loss_tr += loss_per_sample
            (loss_tr/len(lengths)).backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss_tr:.4f}")
        loss_val = 0
        for val_idx in range(len(val_lengths)):
            val_length = val_lengths[val_idx]
            if val_length == 1:
                continue
            elif val_length > 1:
                input_val = val_padded_bags[val_idx, :val_length,:].unsqueeze(0).permute(0, 2, 1)
                cls_val, _, _, _, _ = pn_cls(enc(input_val.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
                pred_label_val = transformation_function(cls_val)
                true_label_val = val_labels[val_idx]
                val_loss_per_sample = criterion(pred_label_val, torch.max(true_label_val.reshape(-1,class_num),1)[1])
                loss_val += val_loss_per_sample
        loss_val_avg = loss_val/len(val_lengths)
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {loss_val_avg:.4f}")
        if loss_val_avg < best_val_loss:
            best_val_loss = loss_val_avg
            torch.save(pn_cls, os.path.join(output_dir, "CLASSIFIER_" + data_source + '_' + str(SEED) + ".pt"))
            torch.save(enc, os.path.join(output_dir, "ENCODER_" + data_source + '_' + str(SEED) + ".pt"))
            patience_counter = 0
            print(f"Saving the best model with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    pn_cls_checkpoint = torch.load(os.path.join(output_dir, "CLASSIFIER_" + data_source + '_' + str(SEED) + ".pt"))
    enc_checkpoint = torch.load(os.path.join(output_dir, "ENCODER_" + data_source + '_' + str(SEED) + ".pt"))
    # Validation metrics pair the saved classifier with the in-memory encoder `enc` (weights from
    # the last epoch trained), not with enc_checkpoint; the training-set scoring below uses both
    # checkpoints.
    pred_label_val_list = []
    true_label_val_list = []
    loss_val = 0
    for val_idx in range(len(val_lengths)):
            val_length = val_lengths[val_idx]
            if val_length == 1:
                continue
            if val_length > 1:
                input_val = val_padded_bags[val_idx, :val_length,:].unsqueeze(0).permute(0, 2, 1)
                cls_val, _, _, _, _ = pn_cls_checkpoint(enc(input_val.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
                pred_label_val = transformation_function(cls_val)
                true_label_val = val_labels[val_idx]
                pred_label_val_list.append(pred_label_val.detach().cpu().numpy())
                true_label_val_list.append(true_label_val.detach().cpu().numpy())
                val_loss_per_sample = criterion(pred_label_val, torch.max(true_label_val.reshape(-1,class_num),1)[1])
                loss_val += val_loss_per_sample
    true_label_val_df = pd.DataFrame(np.vstack(true_label_val_list), columns=Ys_df.columns)
    pred_label_val_df = pd.DataFrame(np.vstack(pred_label_val_list), columns=Ys_df.columns)
    true_label_val_df = true_label_val_df.sort_values(by=[metadata[task_key][0]], ascending=False)
    pred_label_val_df = pred_label_val_df.loc[true_label_val_df.index]
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(pred_label_val_df, cmap="coolwarm", cbar=True)
    plt.show()
    fig.savefig(os.path.join(output_dir, "pred_label_val_" + data_source + '_' + str(SEED) + ".png"))
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(true_label_val_df, cmap="coolwarm", cbar=True)
    plt.show()
    fig.savefig(os.path.join(output_dir, "true_label_val_" + data_source + '_' + str(SEED) + ".png"))
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import f1_score
    auROC_val = roc_auc_score(true_label_val_df, pred_label_val_df)
    pred_label_val_df_ = pred_label_val_df.copy()
    # One-hot of each donor's arg-max class
    pred_label_val_df_ = pred_label_val_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
    macroF1_val = f1_score(true_label_val_df, pred_label_val_df_, average='macro')
    device = 'cpu'
    def MIL_Collate_fn(batch):
        """Zero-pad bags to the longest in the batch and move them to `device`.

        Returns (padded_bags, labels, lengths, metadata_idx); lengths are the bag sizes.
        """
        bags, labels, ins, metadata_idx= zip(*batch)
        lengths = ins
        max_length = max(ins)
        padded_bags = torch.zeros((len(bags), max_length, bags[0].shape[1]), dtype=torch.float32)
        for i, bag in enumerate(bags):
            padded_bags[i, :len(bag)] = bag
        labels = torch.stack(labels)
        return padded_bags.to(device), labels.to(device), lengths, metadata_idx
    # Re-score the training donors in one CPU batch with both saved checkpoints to collect donor
    # predictions and per-cell attention weights.
    train_loader_ = DataLoader(train_dataset, batch_size=train_size, shuffle=False, collate_fn=MIL_Collate_fn)
    for tr_padded_bags, tr_labels, tr_lengths, tr_id in train_loader_:
            print(len(tr_lengths))
    pn_cls_checkpoint=pn_cls_checkpoint.to(device)
    enc_checkpoint=enc_checkpoint.to(device)
    pred_label_tr_list = []
    true_label_tr_list = []
    attention_mtx_list = []
    instance_level_list = []
    cell_id_list = []
    loss_tr = 0
    for tr_idx in range(len(tr_lengths)):
            tr_length = tr_lengths[tr_idx]
            if tr_length == 1:
                continue
            elif tr_length > 1:
                input_tr = tr_padded_bags[tr_idx, :tr_length,:].unsqueeze(0).permute(0, 2, 1)
                cls_tr, _, _, attn_weights, _ = pn_cls_checkpoint(enc_checkpoint(input_tr.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
                pred_label_tr = transformation_function(cls_tr)
                true_label_tr = tr_labels[tr_idx]
                pred_label_tr_list.append(pred_label_tr.detach().cpu().numpy())
                true_label_tr_list.append(true_label_tr.detach().cpu().numpy())
                instance_level_list.append(input_tr.squeeze(0).permute(1,0).detach().cpu().numpy())
                attention_mtx_list.append(attn_weights.squeeze(0,1).detach().cpu().numpy())
                loss_per_sample = criterion(pred_label_tr, torch.max(true_label_tr.to(device).reshape(-1,class_num),1)[1])
                loss_tr += loss_per_sample
                cell_id_list.append(tr_id[tr_idx])
    tr_sample_ids = [adata[adata.obs.index==x[0]].obs[sample_key][0] for x in cell_id_list]
    true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=Ys_df.columns, index=tr_sample_ids)
    pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns, index=tr_sample_ids)
    true_label_tr_df = true_label_tr_df.sort_values(by=[metadata[task_key][0]], ascending=False)
    pred_label_tr_df = pred_label_tr_df.loc[true_label_tr_df.index]
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(pred_label_tr_df, cmap="coolwarm", cbar=True)
    plt.show()
    fig.savefig(os.path.join(output_dir, "pred_label_tr_" + data_source + '_' + str(SEED) + ".png"))
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(true_label_tr_df, cmap="coolwarm", cbar=True)
    plt.show()
    fig.savefig(os.path.join(output_dir, "true_label_tr_" + data_source + '_' + str(SEED) + ".png"))
    auROC_tr = roc_auc_score(true_label_tr_df, pred_label_tr_df, average='macro', multi_class='ovr')
    pred_label_tr_df_ = pred_label_tr_df.copy()
    pred_label_tr_df_ = pred_label_tr_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
    macroF1_tr = f1_score(true_label_tr_df, pred_label_tr_df_, average='macro')
    print(sample_key, '\n',
        "auROC_tr", auROC_tr,'\n',
        "auROC_val", auROC_val,'\n',
        "macroF1_tr", macroF1_tr,'\n',
        "macroF1_val", macroF1_val)
    res_df = pd.DataFrame(
                {
                    'sample_key': sample_key,
                    'auROC_tr': auROC_tr,
                    'macroF1_tr': macroF1_tr,
                    'auROC_val': auROC_val,
                    'macroF1_val': macroF1_val,
                    'SEED': SEED
                }, index=[0]).T
    res_df.to_csv(os.path.join(output_dir, "RES_STAT_" + data_source + '_' + str(SEED) + ".csv"))
    true_label_tr_df.to_csv(os.path.join(output_dir, "true_label_tr_" + data_source + '_' + str(SEED) + ".csv"))
    pred_label_tr_df.to_csv(os.path.join(output_dir, "pred_label_tr_" + data_source + '_' + str(SEED) + ".csv"))
    pred_label_tr_df_.to_csv(os.path.join(output_dir, "pred_label_tr_" + data_source + '_' + str(SEED) + "_.csv"))
    # Training donors whose arg-max prediction is wrong
    wrong_preds = pred_label_tr_df_.idxmax(axis=1)[pred_label_tr_df_.idxmax(axis=1) != true_label_tr_df.idxmax(axis=1)].index.tolist()
    instance_mtx = np.vstack(instance_level_list)
    cell_id_list = [i for sublist in cell_id_list for i in sublist]
    cell_id_df = pd.DataFrame([i for i in cell_id_list], columns=["cell_id"])
    metadata_tr = metadata.loc[cell_id_df.cell_id]
    from scipy.stats import zscore
    attention_mtx_raw = np.concatenate(attention_mtx_list, axis=0)
    attention_mtx_raw_df = pd.DataFrame(attention_mtx_raw, columns=["attention_score_raw"], index=cell_id_df.cell_id)
    # Attention z-scored within each donor.
    attention_mtx_list_norm = [zscore(i) for i in attention_mtx_list]
    attention_mtx = np.concatenate(attention_mtx_list_norm, axis=0)
    # Attention x number of cells in the bag, so 1.0 is the uniform-attention level for that donor.
    attention_mtx_list_norm_cellnum = [i*len(i) for i in attention_mtx_list]
    attention_mtx_norm_cellnum = np.concatenate(attention_mtx_list_norm_cellnum, axis=0)
    attention_mtx_raw_df['attention_score_zscore'] = attention_mtx.tolist()
    attention_mtx_raw_df['attention_score_norm_cellnum'] = attention_mtx_norm_cellnum.tolist()
    attention_mtx_raw_df['attention_score_zscore_clip'] = attention_mtx_raw_df['attention_score_zscore'].apply(lambda x: min(x, ATTENTION_CLIP_MAX))
    attention_mtx_raw_df['attention_score_norm_cellnum_clip'] = attention_mtx_raw_df['attention_score_norm_cellnum'].apply(lambda x: min(x, ATTENTION_CLIP_MAX))
    metadata_tr = pd.concat([metadata_tr,attention_mtx_raw_df], axis=1)
    adata_tr = adata[metadata_tr.index.tolist()]
    adata_tr.obs = metadata_tr.copy()
    adata_tr.obs.index = adata_tr.obs.index.tolist()
    adata_tr.obsm['X_emb'] = adata_tr.obsm['X_sciMultiLaMA']
    sc.pp.neighbors(adata_tr, n_neighbors=25, use_rep='X_emb')
    sc.tl.umap(adata_tr)
    sc.pl.umap(
        adata_tr,
        color=['Supertype'],
        legend_loc='right margin',
        palette='tab10',
        ncols=1,
        wspace=0.5,
        save='_supertype_' + data_source + '_' + str(SEED) + '.png',
    )
    adata_tr.write_h5ad(os.path.join(output_dir, "Adata_Tr_" + data_source + '_' + str(SEED) + ".h5ad"))
    sc.pl.umap(adata_tr, color=['attention_score_norm_cellnum_clip'],
                ncols=1, wspace=0.5, save='_normed_atten_' + data_source + '_' + str(SEED) + '.png')
    sc.pl.umap(adata_tr, color=['attention_score_zscore_clip'],
                ncols=1, wspace=0.5, save='_zscored_atten_' + data_source + '_' + str(SEED) + '.png')
    sc.pl.umap(adata_tr, color=[sample_key, task_key, cov_key, ct_key_fine], legend_loc='on data',
                    ncols=2, wspace=0.5, save='_cov_' + data_source + '_' + str(SEED) + '.png')
    adata_tr_0 = adata_tr[adata_tr.obs[task_key] != metadata[task_key][0]]
    adata_tr_1 = adata_tr[adata_tr.obs[task_key] == metadata[task_key][0]]
    sc.pl.umap(adata_tr_0, color=['attention_score_norm_cellnum_clip'], 
            title = adata_tr_0.obs[task_key][0],
                ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_0.obs[task_key][0] + data_source + '_' + str(SEED) + '.png')
    sc.pl.umap(adata_tr_1, color=['attention_score_norm_cellnum_clip'], 
            title = adata_tr_1.obs[task_key][0],
                ncols=1, wspace=0.5, save='_normed_atten_' + adata_tr_1.obs[task_key][0] + data_source + '_' + str(SEED) + '.png')
    donors = adata_tr_0.obs[sample_key].unique()
    n = len(donors)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    for ax, donor in zip(axes.flat, donors):
        pred_label = ' Right Pred'
        if donor in wrong_preds:
                print(donor)
                pred_label = ' Wrong Pred'
                sc.pl.umap(adata_tr_0[adata_tr_0.obs[sample_key] == donor],
                    color=['attention_score_norm_cellnum_clip'],
                    ax=ax, palette='plasma',
                    show=False,
                    title=donor + pred_label)
        else:
            sc.pl.umap(adata_tr_0[adata_tr_0.obs[sample_key] == donor],
                    color=['attention_score_norm_cellnum_clip'],
                    ax=ax,
                    show=False,
                    title=donor + pred_label)
    for i in range(n, len(axes.flat)):
        fig.delaxes(axes.flat[i])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'umap_normed_atten_' + adata_tr_0.obs[task_key][0] + data_source + '_' + str(SEED) + '_all_donors.png'))
    donors = adata_tr_1.obs[sample_key].unique()
    n = len(donors)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    for ax, donor in zip(axes.flat, donors):
        pred_label = ' Right Pred'
        if donor in wrong_preds:
                print(donor)
                pred_label = ' Wrong Pred'
                sc.pl.umap(adata_tr_1[adata_tr_1.obs[sample_key] == donor],
                    color=['attention_score_norm_cellnum_clip'],
                    ax=ax, palette='plasma',
                    show=False,
                    title=donor + pred_label)
        else:
            sc.pl.umap(adata_tr_1[adata_tr_1.obs[sample_key] == donor],
                    color=['attention_score_norm_cellnum_clip'],
                    ax=ax,
                    show=False,
                    title=donor + pred_label)
    for i in range(n, len(axes.flat)):
        fig.delaxes(axes.flat[i])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'umap_normed_atten_' + adata_tr_1.obs[task_key][0] + data_source + '_' + str(SEED) + '_all_donors.png'))
    # Score every donor (train + validation) with the saved checkpoints, group donors by predicted
    # class, and mark each donor's top TOP_FRAC cells by raw attention on the all-donor UMAP.
    if MAKE_ALL_DONOR_TOP10_PLOTS:
        if POS_LABEL not in Ys_df.columns:
            raise ValueError(f"POS_LABEL={POS_LABEL} not found in {Ys_df.columns.tolist()}")
        neg_labels = [c for c in Ys_df.columns if c != POS_LABEL]
        if len(neg_labels) != 1:
            raise ValueError(f"Set POS_LABEL so the remaining label is unambiguous: {Ys_df.columns.tolist()}")
        neg_label = neg_labels[0]
        full_loader_ = DataLoader(mil_dataset, batch_size=len(mil_dataset), shuffle=False, collate_fn=MIL_Collate_fn)
        for full_padded_bags, full_labels, full_lengths, full_id in full_loader_:
            full_ids = full_id
            break
        pred_label_full_list = []
        full_attention_list = []
        full_cell_id_list = []
        for full_idx in range(len(full_lengths)):
            full_length = full_lengths[full_idx]
            if full_length <= 1:
                continue
            input_full = full_padded_bags[full_idx, :full_length, :].unsqueeze(0).permute(0, 2, 1)
            cls_full, _, _, attn_weights, _ = pn_cls_checkpoint(
                enc_checkpoint(input_full.squeeze(0).transpose(0, 1), None).transpose(0, 1).unsqueeze(0)
            )
            pred_label_full_list.append(transformation_function(cls_full).detach().cpu().numpy())
            full_attention_list.append(attn_weights.squeeze(0, 1).detach().cpu().numpy())
            full_cell_id_list.append(full_ids[full_idx])
        full_cell_id_list = [i for sublist in full_cell_id_list for i in sublist]
        full_attention_raw = np.concatenate(full_attention_list, axis=0)
        full_attention_df = pd.DataFrame(
            full_attention_raw, columns=["attention_score_raw"], index=full_cell_id_list
        )
        adata_full = adata.copy()
        adata_full.obs = adata_full.obs.join(full_attention_df, how="left")
        if "X_umap" not in adata_full.obsm:
            sc.pp.neighbors(adata_full, n_neighbors=25, use_rep="X_sciMultiLaMA")
            sc.tl.umap(adata_full, random_state=ALL_DONOR_UMAP_SEED)
        full_sample_ids = [adata_full.obs.loc[x[0], sample_key] for x in full_ids]
        pred_label_full_df = pd.DataFrame(
            np.vstack(pred_label_full_list), columns=Ys_df.columns, index=full_sample_ids
        )
        pred_label_full_df = pred_label_full_df[~pred_label_full_df.index.duplicated(keep="first")]
        pred_label_full = pred_label_full_df.idxmax(axis=1)
        ad_donors = pred_label_full.index[pred_label_full == POS_LABEL]
        ctrl_donors = pred_label_full.index[pred_label_full == neg_label]

        def plot_top10_for_donors(donor_ids, tag):
            """Mark each donor's top TOP_FRAC cells by raw attention on the all-donor UMAP and save it."""
            top_mask = pd.Series(False, index=adata_full.obs.index)
            for donor in donor_ids:
                donor_mask = adata_full.obs[sample_key] == donor
                scores = adata_full.obs.loc[donor_mask, "attention_score_raw"].to_numpy()
                if scores.size == 0:
                    continue
                thresh = np.quantile(scores, 1 - TOP_FRAC)
                top_mask.loc[donor_mask] = adata_full.obs.loc[donor_mask, "attention_score_raw"] >= thresh
            adata_full.obs["top_cells"] = "Background"
            top_label = f"Top {int(TOP_FRAC * 100)}%"
            adata_full.obs.loc[top_mask, "top_cells"] = top_label
            sc.pl.umap(
                adata_full,
                color="top_cells",
                palette={"Background": "#D3D3D3", top_label: "#E64B35"},
                size=10,
                alpha=0.85,
                show=False,
                title=tag,
                save=f"_top{int(TOP_FRAC * 100)}_{tag}_MTG_{SEED}.png",
            )

        plot_top10_for_donors(ad_donors, f"pred_{POS_LABEL}")
        plot_top10_for_donors(ctrl_donors, f"pred_{neg_label}")
