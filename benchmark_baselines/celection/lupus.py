"""CELECTION, lupus SLE vs healthy ("Lupus" in Fig. 2 / Table S3).

The reference script for the CELECTION benchmark runs. covid.py, lupus_race.py, cardio.py and
cardio_biosample.py repeat this loop and only note what they change.

Each donor is a bag of cells (obs 'ind_cov', 169 donors in $DATA_DIR/lupus/lupus_processed.h5ad),
labelled by obs 'disease_cov' (healthy / sle). The model code is bioPointNet_Apr2025.py from
$BIOPOINTNET_ROOT (default "bioPointNet_dev"): a CellEncoder on every cell (genes -> 256 -> 32,
LayerNorm, dropout 0.1, extra 32 -> 32 linear layer) feeding a PointNetClassHead with gated-attention
pooling.

Protocol: 10 seeds x the 5 predefined folds of lupus_5folds.csv ('samples' = bag id, 'fold' = 1-5), so
50 runs. The test fold is held out and the other bags are split 80/20 into train / validation by
random_split seeded with the seed. Adam (lr 1e-4) on encoder and head, 5 bags per step, at most 200
epochs, early stopping after 20 epochs without a lower validation loss. The loss is CrossEntropyLoss on
the softmax output, so softmax is applied twice. Single-cell bags are skipped throughout, and neither
module is ever put in eval mode (dropout stays on when scoring). The head and encoder of the
lowest-validation-loss epoch are saved and score the train and test bags; the validation bags get the
saved head on top of the last-epoch encoder.

Metrics per split: auROC (roc_auc_score on the one-hot labels, macro), macroF1 (argmax class) and
macroAUPRC (mean per-class area under the precision-recall curve).

Writes to $RESULTS_DIR/R2-q1/Results/CELECTION/Lupus/, with <task> = lupus_cls:
  best_pn_<task>_<fold>_seed_<seed>.pt        saved head (whole module)
  best_enc_<task>_<fold>_seed_<seed>.pt       saved encoder (whole module)
  <task>_results_fold_<fold>_seed_<seed>.csv  metrics of one run
  <task>_results_all_folds.pth                all per-run tables, keyed 'fold_<fold>_seed_<seed>'
  <task>_metrics_all.csv                      one row per run and split; read by summarize.py
"""
# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.metrics import precision_recall_curve, auc

# Folder containing bioPointNet_Apr2025.py (the CELECTION model code); see README.
BIOPOINTNET_ROOT = os.environ.get("BIOPOINTNET_ROOT", "bioPointNet_dev")
if BIOPOINTNET_ROOT not in sys.path:
    sys.path.append(BIOPOINTNET_ROOT)

from bioPointNet_Apr2025 import *

TRAIN_FRACTION = 0.8  # the rest of train+val is the validation set


path = f"{DATA_DIR}/lupus/"
adata = sc.read_h5ad(path + "lupus_processed.h5ad")
# Label and bag counts; these bare expressions only show output when run interactively.
adata.obs.disease_cov.value_counts()
adata.obs.pop_cov.value_counts()
adata_full = adata.copy()
adata.obs.ind_cov.value_counts()
adata.obs[['disease_cov','ind_cov']].value_counts().sort_index()

task_name = 'lupus_cls'
sample_key = 'ind_cov'
task_key = 'disease_cov'
class_num = 2

folds = pd.read_csv(path + "lupus_5folds.csv", index_col=0)

save_dir = f"{RESULTS_DIR}/R2-q1/Results/CELECTION/Lupus"
os.makedirs(save_dir, exist_ok=True)


def compute_macro_auprc(y_true_df, y_pred_df):
    """Mean PR-curve area over the one-hot class columns, skipping single-valued ones (NaN if none)."""
    scores = []
    for c in y_true_df.columns:
        if len(np.unique(y_true_df[c])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true_df[c], y_pred_df[c])
        scores.append(auc(recall, precision))
    return float(np.mean(scores)) if scores else float("nan")


res_stat_dict = {}
metrics_rows = []
for SEED in range(10):
    for fold in [1,2,3,4,5]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds_trainval = folds[folds['fold'] != fold]
        folds_test = folds[folds['fold'] == fold]
        adata = adata_full[adata_full.obs[sample_key].isin(folds_trainval['samples'])]
        adata_test = adata_full[adata_full.obs[sample_key].isin(folds_test['samples'])]
        metadata = adata.obs.copy()
        Ns = adata.X.copy() # sparse matrix
        Ns_df = pd.DataFrame(Ns.todense(), index=metadata.index.tolist(), columns=adata.var.index.tolist())
        Ys_df = pd.get_dummies(adata.obs[task_key].copy()).astype(int)
        # Xs[i]: (n_cells_i, n_genes); Ys[i]: one-hot bag label; ins[i]: n_cells_i.
        Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata, bag_column=sample_key)
        print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
        mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
        metadata_test = adata_test.obs.copy()
        Ns_test = adata_test.X.copy() # sparse matrix
        Ns_df_test = pd.DataFrame(Ns_test.todense(), index=metadata_test.index.tolist(), columns=adata_test.var.index.tolist())
        Ys_df_test = pd.get_dummies(adata_test.obs[task_key].copy()).astype(int)
        Xs_test, Ys_test, ins_test, meta_ids_test = Create_MIL_Dataset(ins_df=Ns_df_test, label_df=Ys_df_test, metadata=metadata_test, bag_column=sample_key)
        print(f"Number of bags: {len(Xs_test)}", f"Number of labels: {Ys_test.shape}", f"Number of max instances: {max(ins_test)}")
        mil_dataset_test = MILDataset(Xs_test, Ys_test, ins_test, meta_ids_test)
        print(f"Training set size: {len(mil_dataset)}", f"Testing set size: {len(mil_dataset_test)}", '\n')
        train_size = int(TRAIN_FRACTION * len(mil_dataset))
        val_size = len(mil_dataset) - train_size
        torch.manual_seed(SEED)
        set_seed(SEED)
        train_dataset, val_dataset = random_split(mil_dataset, [train_size, val_size])
        print(f"Training set size: {len(train_dataset)}", f"Validation set size: {len(val_dataset)}")
        batch_size = 5
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=MIL_Collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=val_size, shuffle=False, collate_fn=MIL_Collate_fn)
        set_seed(SEED)
        enc = CellEncoder(input_dim=Ns.shape[1], input_batch_num=0, hidden_layer=[256, 32],
                        layernorm=True, activation=nn.ReLU(), batchnorm=False, dropout_rate=0.1,
                        add_linear_layer=True, clip_threshold=None)
        enc.apply(init_weights)
        enc.to(device)
        pn_cls = PointNetClassHead(input_dim=32, k=class_num, global_features=128, attention_dim=32,
                                   agg_method="gated_attention")
        pn_cls.apply(init_weights)
        pn_cls.to(device)
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(list(pn_cls.parameters())+list(enc.parameters()), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        transformation_function = nn.Softmax(dim=1)
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        best_model_state = None
        epochs = 200
        # One batch with every validation bag, zero-padded to (n_val_bags, max_cells, n_genes);
        # val_lengths holds the real cell counts.
        for val_padded_bags, val_labels, val_lengths, val_id in val_loader:
                print(len(val_lengths))
        # train_model / eval_model are never called: encoder and head stay in training mode, so dropout
        # is also on when the validation and test bags are scored.
        def train_model(model):
            model.train()
        def eval_model(model):
            model.eval()
        import copy
        for epoch in range(epochs):
            for padded_bags, labels, lengths, _ in train_loader:
                optimizer.zero_grad()
                loss_tr = 0
                # One bag at a time with the padding stripped; single-cell bags are skipped.
                for idx in range(len(lengths)):
                    length = lengths[idx]
                    if length == 1:
                        continue
                    elif length > 1:
                        # input_tr is (1, n_genes, n_cells); enc runs per cell on (n_cells, n_genes),
                        # and the head takes (1, 32, n_cells).
                        input_tr = padded_bags[idx, :length,:].unsqueeze(0).permute(0, 2, 1)
                        cls, embedding, global_features, attn_weights, crit_idxs = pn_cls(enc(input_tr.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
                        pred_label = transformation_function(cls)
                        true_label = labels[idx]
                        # CrossEntropyLoss gets softmax probabilities, not logits (softmax applied twice).
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
                torch.save(pn_cls, f"{save_dir}/best_pn_{task_name}_{fold}_seed_{SEED}.pt")
                torch.save(enc, f"{save_dir}/best_enc_{task_name}_{fold}_seed_{SEED}.pt")
                patience_counter = 0
                print(f"Saving the best model with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        pn_cls_checkpoint = torch.load(f"{save_dir}/best_pn_{task_name}_{fold}_seed_{SEED}.pt")
        enc_checkpoint = torch.load(f"{save_dir}/best_enc_{task_name}_{fold}_seed_{SEED}.pt")
        # Best head on the last-epoch encoder `enc`; the train and test bags below get enc_checkpoint.
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
        # y_true is one-hot, so roc_auc_score averages per-class AUROCs; F1 takes the argmax class.
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import f1_score
        auROC_val = roc_auc_score(true_label_val_df, pred_label_val_df, average='macro', multi_class='ovr')
        pred_label_val_df_ = pred_label_val_df.copy()
        pred_label_val_df_ = pred_label_val_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
        macroF1_val = f1_score(true_label_val_df, pred_label_val_df_, average='macro')
        macroAUPRC_val = compute_macro_auprc(true_label_val_df, pred_label_val_df)
        # Shadows the imported MIL_Collate_fn so the full train and test sets, one batch each, are scored
        # on the CPU.
        device = 'cpu'
        def MIL_Collate_fn(batch):
            """Zero-pad to the largest bag; returns (padded_bags, labels, lengths, cell ids)."""
            bags, labels, ins, metadata_idx= zip(*batch)
            lengths = ins
            max_length = max(ins)
            padded_bags = torch.zeros((len(bags), max_length, bags[0].shape[1]), dtype=torch.float32)
            for i, bag in enumerate(bags):
                padded_bags[i, :len(bag)] = bag
            labels = torch.stack(labels)
            return padded_bags.to(device), labels.to(device), lengths, metadata_idx
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
        true_label_tr_df = pd.DataFrame(np.vstack(true_label_tr_list), columns=Ys_df.columns)
        pred_label_tr_df = pd.DataFrame(np.vstack(pred_label_tr_list), columns=Ys_df.columns)
        auROC_tr = roc_auc_score(true_label_tr_df, pred_label_tr_df, average='macro', multi_class='ovr')
        pred_label_tr_df_ = pred_label_tr_df.copy()
        pred_label_tr_df_ = pred_label_tr_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
        macroF1_tr = f1_score(true_label_tr_df, pred_label_tr_df_, average='macro')
        macroAUPRC_tr = compute_macro_auprc(true_label_tr_df, pred_label_tr_df)
        test_loader = DataLoader(mil_dataset_test, batch_size=len(mil_dataset_test), shuffle=False, collate_fn=MIL_Collate_fn)
        for te_padded_bags, te_labels, te_lengths, te_id in test_loader:
                print(len(te_lengths))
        pred_label_test_list = []
        true_label_test_list = []
        attention_mtx_list_test = []
        instance_level_list_test = []
        cell_id_list_test = []
        loss_test = 0
        for test_idx in range(len(te_lengths)):
                    test_length = te_lengths[test_idx]
                    if test_length == 1:
                        continue
                    elif test_length > 1:
                        input_test = te_padded_bags[test_idx, :test_length,:].unsqueeze(0).permute(0, 2, 1)
                        cls_test, _, _, attn_weights, _ = pn_cls_checkpoint(enc_checkpoint(input_test.squeeze(0).transpose(0,1), None).transpose(0,1).unsqueeze(0))
                        pred_label_test = transformation_function(cls_test)
                        true_label_test = te_labels[test_idx]
                        pred_label_test_list.append(pred_label_test.detach().cpu().numpy())
                        true_label_test_list.append(true_label_test.detach().cpu().numpy())
                        instance_level_list_test.append(input_test.squeeze(0).permute(1,0).detach().cpu().numpy())
                        attention_mtx_list_test.append(attn_weights.squeeze(0,1).detach().cpu().numpy())
                        loss_per_sample = criterion(pred_label_test, torch.max(true_label_test.to(device).reshape(-1,class_num),1)[1])
                        loss_test += loss_per_sample
                        cell_id_list_test.append(te_id[test_idx])
        true_label_test_df = pd.DataFrame(np.vstack(true_label_test_list), columns=Ys_df.columns)
        pred_label_test_df = pd.DataFrame(np.vstack(pred_label_test_list), columns=Ys_df.columns)
        auROC_test = roc_auc_score(true_label_test_df, pred_label_test_df, average='macro', multi_class='ovr')
        pred_label_test_df_ = pred_label_test_df.copy()
        pred_label_test_df_ = pred_label_test_df_.apply(lambda x: x == x.max(), axis=1).astype(int)
        macroF1_test = f1_score(true_label_test_df, pred_label_test_df_, average='macro')
        macroAUPRC_test = compute_macro_auprc(true_label_test_df, pred_label_test_df)
        print(sample_key, '\n',
            "auROC_tr", auROC_tr,'\n',
            "auROC_val", auROC_val,'\n',
            "macroF1_tr", macroF1_tr,'\n',
            "macroF1_val", macroF1_val,'\n',
            "macroAUPRC_tr", macroAUPRC_tr,'\n',
            "macroAUPRC_val", macroAUPRC_val,'\n',
            "auROC_test", auROC_test,'\n',
            "macroF1_test", macroF1_test,'\n',
            "macroAUPRC_test", macroAUPRC_test,'\n',
            "fold", fold, '\n',
            'seed', SEED
            )
        res_df = pd.DataFrame(
            {
                'sample_key': sample_key,
                'auROC_tr': auROC_tr,
                'macroF1_tr': macroF1_tr,
                'macroAUPRC_tr': macroAUPRC_tr,
                'auROC_val': auROC_val,
                'macroF1_val': macroF1_val,
                'macroAUPRC_val': macroAUPRC_val,
                'auROC_test': auROC_test,
                'macroF1_test': macroF1_test,
                'macroAUPRC_test': macroAUPRC_test,
                'fold': fold
            }, index=[0]
        )
        res_df = res_df.T
        res_df.columns = [task_name]
        res_df.to_csv(f"{save_dir}/{task_name}_results_fold_{fold}_seed_{SEED}.csv")
        res_stat_dict['fold_' + str(fold) + '_seed_' + str(SEED)] = res_df
        # One row per split for <task>_metrics_all.csv.
        for split_name, metrics in [
            ("train", (auROC_tr, macroF1_tr, macroAUPRC_tr)),
            ("val", (auROC_val, macroF1_val, macroAUPRC_val)),
            ("test", (auROC_test, macroF1_test, macroAUPRC_test)),
        ]:
            metrics_rows.append(
                {
                    "model": "CELECTION",
                    "seed": SEED,
                    "fold": fold,
                    "split": split_name,
                    "auROC": metrics[0],
                    "macroF1": metrics[1],
                    "macroAUPRC": metrics[2],
                }
            )


torch.save(res_stat_dict, f"{save_dir}/{task_name}_results_all_folds.pth")


# Per-run test metrics. The .mean() lines below are leftovers from interactive use and print
# nothing in a script.
res_stat_dict.keys()


res_test_df = pd.DataFrame(index=list(res_stat_dict.keys()),
                           columns=['auROC_test','macroF1_test','macroAUPRC_test'])


for i in list(res_stat_dict.keys()):
     print(i)
     res_test_df.loc[i,'auROC_test'] = res_stat_dict[i].loc['auROC_test'].values
     res_test_df.loc[i,'macroF1_test'] = res_stat_dict[i].loc['macroF1_test'].values
     res_test_df.loc[i,'macroAUPRC_test'] = res_stat_dict[i].loc['macroAUPRC_test'].values


res_test_df.mean()


# Read by summarize.py.
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(f"{save_dir}/{task_name}_metrics_all.csv", index=False)


for seed in range(10):
    print(seed)
    res_test_df[res_test_df.index.str.contains('seed_' + str(seed))].mean()
    print('\n')


for fold in [1,2,3,4,5]:
    res_test_df[res_test_df.index.str.contains('fold_' + str(fold))].mean()
    print('\n')
