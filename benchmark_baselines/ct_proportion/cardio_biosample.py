"""Same as lupus.py for Cardio at biosample level: one sample per biosample (80 biosamples from 42 donors: 20 DCM,
30 HCM, 30 NF).

Sample = biosample_id, folds from $DATA_DIR/cardio/cardio_biosample_id_5folds.csv; otherwise as cardio.py,
including compute_metrics_multiclass. Unlike the other two baselines, both models are fit at this level. Task name
cardio_cls_biosample; outputs in $RESULTS_DIR/R2-q1/Results/proportion/CardioBiosample/.
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
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from torch.utils.data import Dataset

path = f"{DATA_DIR}/cardio/"
adata = sc.read_h5ad(os.path.join(path, "cardio_processed.h5ad"))
# Bare expressions left from interactive exploration; nothing below uses their values.
adata.obs.disease.value_counts()
adata.obs.biosample_id.value_counts()
adata.obs.donor_id.value_counts()
adata.obs[['disease','biosample_id']].value_counts().sort_index()
adata.obs['cell_type_leiden0.6'].value_counts()
adata_full = adata.copy()


task_name = 'cardio_cls_biosample'
sample_key = 'biosample_id'
task_key = 'disease'
ct_key = 'cell_type_leiden0.6'
class_num = 3

folds = pd.read_csv(os.path.join(path, "cardio_biosample_id_5folds.csv"), index_col=0)

save_dir = f"{RESULTS_DIR}/R2-q1/Results/proportion/CardioBiosample"
os.makedirs(save_dir, exist_ok=True)


def compute_macro_auroc(y_true_onehot, y_pred_proba):
    """Mean one-vs-rest AUROC over the label columns; single-valued columns are skipped (NaN if none remain)."""
    scores = []
    for c in range(y_true_onehot.shape[1]):
        if len(np.unique(y_true_onehot[:, c])) < 2:
            continue
        scores.append(roc_auc_score(y_true_onehot[:, c], y_pred_proba[:, c]))
    return float(np.mean(scores)) if scores else float("nan")


def compute_macro_auprc(y_true_onehot, y_pred_proba):
    """As compute_macro_auroc, for the trapezoidal area under the precision-recall curve."""
    scores = []
    for c in range(y_true_onehot.shape[1]):
        if len(np.unique(y_true_onehot[:, c])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, c], y_pred_proba[:, c])
        scores.append(auc(recall, precision))
    return float(np.mean(scores)) if scores else float("nan")


def compute_metrics_multiclass(y_true_onehot, y_pred_proba, y_pred_logit):
    """auROC, macroAUPRC and macroF1 of one split; macro-F1 compares argmax classes, not indicator matrices."""
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_logit, axis=1)
    return {
        "auROC": compute_macro_auroc(y_true_onehot, y_pred_proba),
        "macroAUPRC": compute_macro_auprc(y_true_onehot, y_pred_proba),
        "macroF1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


project_pc = False
res_stat_dict_RF = {}
res_stat_dict_LogisticRegression = {}
metrics_rows = []
for SEED in range(10):
    for fold in [1,2,3,4,5]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds_trainval = folds[folds['fold'] != fold]
        folds_test = folds[folds['fold'] == fold]
        adata_trainval = adata[adata.obs[sample_key].isin(folds_trainval.samples.tolist())]
        adata_test = adata[adata.obs[sample_key].isin(folds_test.samples.tolist())]
        # Labels come from the folds-file index. Train+val and test are one-hot encoded together, so both get the
        # same columns, then split back by position; rows follow folds_*['samples'].
        y_sample_label_trainval = folds_trainval.index
        y_sample_label_test = folds_test.index
        y_sample_label_all = np.concatenate([y_sample_label_trainval, y_sample_label_test])
        y_sample_label_all_onehot = pd.get_dummies(y_sample_label_all).astype(int)
        y_sample_label_trainval_onehot = y_sample_label_all_onehot.iloc[:len(y_sample_label_trainval)]
        y_sample_label_test_onehot = y_sample_label_all_onehot.iloc[len(y_sample_label_trainval):]
        celltype_list = adata.obs[ct_key].unique().tolist()
        num_genes = adata.shape[1]
        num_celltypes = len(celltype_list)
        # Proportions use all cells of every sample in adata, so they are the same in every fold.
        celltype_proportions = {}
        for sample in adata.obs[sample_key].unique():
            adata_sample = adata[adata.obs[sample_key] == sample]
            celltype_counts = adata_sample.obs[ct_key].value_counts()
            total_cells = celltype_counts.sum()
            proportions = celltype_counts / total_cells if total_cells > 0 else np.zeros(len(celltype_counts))
            celltype_proportions[sample] = proportions
        # One row per sample, one column per cell type; cell types absent from a sample get 0
        celltype_proportions_df = pd.DataFrame(celltype_proportions).T.fillna(0)
        # Rows are re-ordered to match folds_*['samples'], i.e. the label order above.
        X_ct_prop_trainval = celltype_proportions_df.loc[folds_trainval['samples']]
        X_ct_prop_test = celltype_proportions_df.loc[folds_test['samples']]
        # The validation part is only scored; nothing is tuned on it.
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_ct_prop_trainval, y_sample_label_trainval_onehot, test_size=0.2, random_state=SEED)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=SEED)
        clf.fit(X_train, y_train.values.argmax(1))
        y_pred_train = clf.predict_proba(X_train)
        y_pred_val = clf.predict_proba(X_val)
        y_pred_test = clf.predict_proba(X_ct_prop_test)
        # Despite the name, *_logit is the 0/1 indicator of the most probable class (ties give several 1s).
        y_pred_train_logit = pd.DataFrame(y_pred_train).apply(lambda x: x == x.max(), axis=1).astype(int)
        y_pred_val_logit = pd.DataFrame(y_pred_val).apply(lambda x: x == x.max(), axis=1).astype(int)
        y_pred_test_logit = pd.DataFrame(y_pred_test).apply(lambda x: x == x.max(), axis=1).astype(int)
        metrics_tr = compute_metrics_multiclass(y_train.values, y_pred_train, y_pred_train_logit.values)
        metrics_val = compute_metrics_multiclass(y_val.values, y_pred_val, y_pred_val_logit.values)
        metrics_te = compute_metrics_multiclass(
            y_sample_label_test_onehot.values, y_pred_test, y_pred_test_logit.values
        )
        print(sample_key, '\n',
                                    "auROC_tr", metrics_tr["auROC"],'\n',
                                    "macroF1_tr", metrics_tr["macroF1"],'\n',
                                    "macroAUPRC_tr", metrics_tr["macroAUPRC"],'\n',
                                    "auROC_val", metrics_val["auROC"],'\n',
                                    "macroF1_val", metrics_val["macroF1"],'\n',
                                    "macroAUPRC_val", metrics_val["macroAUPRC"],'\n',
                                    "auROC_test", metrics_te["auROC"],'\n',
                                    "macroF1_test", metrics_te["macroF1"],'\n',
                                    "macroAUPRC_test", metrics_te["macroAUPRC"],'\n',
                                    "fold", fold, '\n',
                                    'seed', SEED
                                    )
        res_df = pd.DataFrame(
                    {
                        'sample_key': sample_key,
                        'auROC_tr': metrics_tr["auROC"],
                        'macroF1_tr': metrics_tr["macroF1"],
                        'macroAUPRC_tr': metrics_tr["macroAUPRC"],
                        'auROC_val': metrics_val["auROC"],
                        'macroF1_val': metrics_val["macroF1"],
                        'macroAUPRC_val': metrics_val["macroAUPRC"],
                        'auROC_test': metrics_te["auROC"],
                        'macroF1_test': metrics_te["macroF1"],
                        'macroAUPRC_test': metrics_te["macroAUPRC"],
                        'fold': fold
                    }, index=[0]
                )
        res_df = res_df.T
        res_df.columns = [task_name]
        res_df.to_csv(f'{save_dir}/RF_{task_name}_results_fold_{fold}_seed_{SEED}.csv')
        res_stat_dict_RF['fold_' + str(fold) + '_seed_' + str(SEED)] = res_df
        for split_name, metrics in [
            ("train", metrics_tr),
            ("val", metrics_val),
            ("test", metrics_te),
        ]:
            metrics_rows.append(
                {
                    "model": "RF",
                    "seed": SEED,
                    "fold": fold,
                    "split": split_name,
                    "auROC": metrics["auROC"],
                    "macroF1": metrics["macroF1"],
                    "macroAUPRC": metrics["macroAUPRC"],
                }
            )
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.linear_model import LogisticRegression
        clf = OneVsRestClassifier(LogisticRegression(random_state=SEED))
        clf.fit(X_train, y_train.values.argmax(1))
        y_pred_train = clf.predict_proba(X_train)
        y_pred_val = clf.predict_proba(X_val)
        y_pred_test = clf.predict_proba(X_ct_prop_test)
        y_pred_train_logit = pd.DataFrame(y_pred_train).apply(lambda x: x == x.max(), axis=1).astype(int)
        y_pred_val_logit = pd.DataFrame(y_pred_val).apply(lambda x: x == x.max(), axis=1).astype(int)
        y_pred_test_logit = pd.DataFrame(y_pred_test).apply(lambda x: x == x.max(), axis=1).astype(int)
        metrics_tr = compute_metrics_multiclass(y_train.values, y_pred_train, y_pred_train_logit.values)
        metrics_val = compute_metrics_multiclass(y_val.values, y_pred_val, y_pred_val_logit.values)
        metrics_te = compute_metrics_multiclass(
            y_sample_label_test_onehot.values, y_pred_test, y_pred_test_logit.values
        )
        print(sample_key, '\n',
                                    "auROC_tr", metrics_tr["auROC"],'\n',
                                    "macroF1_tr", metrics_tr["macroF1"],'\n',
                                    "macroAUPRC_tr", metrics_tr["macroAUPRC"],'\n',
                                    "auROC_val", metrics_val["auROC"],'\n',
                                    "macroF1_val", metrics_val["macroF1"],'\n',
                                    "macroAUPRC_val", metrics_val["macroAUPRC"],'\n',
                                    "auROC_test", metrics_te["auROC"],'\n',
                                    "macroF1_test", metrics_te["macroF1"],'\n',
                                    "macroAUPRC_test", metrics_te["macroAUPRC"],'\n',
                                    "fold", fold, '\n',
                                    'seed', SEED
                                    )
        res_df = pd.DataFrame(
                    {
                        'sample_key': sample_key,
                        'auROC_tr': metrics_tr["auROC"],
                        'macroF1_tr': metrics_tr["macroF1"],
                        'macroAUPRC_tr': metrics_tr["macroAUPRC"],
                        'auROC_val': metrics_val["auROC"],
                        'macroF1_val': metrics_val["macroF1"],
                        'macroAUPRC_val': metrics_val["macroAUPRC"],
                        'auROC_test': metrics_te["auROC"],
                        'macroF1_test': metrics_te["macroF1"],
                        'macroAUPRC_test': metrics_te["macroAUPRC"],
                        'fold': fold
                    }, index=[0]
                )
        res_df = res_df.T
        res_df.columns = [task_name]
        res_df.to_csv(f'{save_dir}/LogisticRegression_{task_name}_results_fold_{fold}_seed_{SEED}.csv')
        res_stat_dict_LogisticRegression['fold_' + str(fold) + '_seed_' + str(SEED)] = res_df
        for split_name, metrics in [
            ("train", metrics_tr),
            ("val", metrics_val),
            ("test", metrics_te),
        ]:
            metrics_rows.append(
                {
                    "model": "LogisticRegression",
                    "seed": SEED,
                    "fold": fold,
                    "split": split_name,
                    "auROC": metrics["auROC"],
                    "macroF1": metrics["macroF1"],
                    "macroAUPRC": metrics["macroAUPRC"],
                }
            )


torch.save(res_stat_dict_RF, f'{save_dir}/{task_name}_res_stat_dict_RF.pth')
torch.save(res_stat_dict_LogisticRegression, f'{save_dir}/{task_name}_res_stat_dict_LogisticRegression.pth')

res_stat_dict_RF
res_stat_dict_LogisticRegression


res_test_df_RF = pd.DataFrame(index=list(res_stat_dict_RF.keys()),
                           columns=['auROC_test','macroF1_test','macroAUPRC_test'])
for i in list(res_stat_dict_RF.keys()):
     print(i)
     res_test_df_RF.loc[i,'auROC_test'] = res_stat_dict_RF[i].loc['auROC_test'].values
     res_test_df_RF.loc[i,'macroF1_test'] = res_stat_dict_RF[i].loc['macroF1_test'].values
     res_test_df_RF.loc[i,'macroAUPRC_test'] = res_stat_dict_RF[i].loc['macroAUPRC_test'].values


res_test_df_LogisticRegression = pd.DataFrame(index=list(res_stat_dict_LogisticRegression.keys()),
                           columns=['auROC_test','macroF1_test','macroAUPRC_test'])

for i in list(res_stat_dict_LogisticRegression.keys()):
     print(i)
     res_test_df_LogisticRegression.loc[i,'auROC_test'] = res_stat_dict_LogisticRegression[i].loc['auROC_test'].values
     res_test_df_LogisticRegression.loc[i,'macroF1_test'] = res_stat_dict_LogisticRegression[i].loc['macroF1_test'].values
     res_test_df_LogisticRegression.loc[i,'macroAUPRC_test'] = res_stat_dict_LogisticRegression[i].loc['macroAUPRC_test'].values


res_test_df_RF.mean()
res_test_df_RF.mean().to_csv(f'{save_dir}/RF_{task_name}_results_mean.csv')


res_test_df_LogisticRegression.mean()
res_test_df_LogisticRegression.mean().to_csv(f'{save_dir}/LogisticRegression_{task_name}_results_mean.csv')


metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(f"{save_dir}/{task_name}_metrics_all.csv", index=False)
