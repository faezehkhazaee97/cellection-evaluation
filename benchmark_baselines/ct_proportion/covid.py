"""Same as lupus.py for Ziegler COVID-19: case vs control, one sample per donor (58 donors: 37 case, 21 control).

Reads $DATA_DIR/covid/covid_Ziegler_processed.h5ad (sample = donor_id, cell type = celltype) and
$DATA_DIR/covid/covid_5folds.csv (label disease_id: case / control). The obs column disease_id that the script
derives from disease__ontology_label is only inspected; labels come from the folds file. Task name covid_cls;
outputs in $RESULTS_DIR/R2-q1/Results/proportion/Covid/.
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
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from torch.utils.data import Dataset

path = f"{DATA_DIR}/covid/"
adata = sc.read_h5ad(os.path.join(path, "covid_Ziegler_processed.h5ad"))
# Bare expressions left from interactive exploration; nothing below uses their values.
adata.obs.disease__ontology_label.value_counts()
# disease_id: COVID-19 and long COVID-19 -> 'case', everything else (normal, respiratory failure) -> 'control'.
# The classifiers use the labels of the folds file, not this column.
adata.obs['disease_id'] = 'control'
adata.obs.loc[adata.obs['disease__ontology_label'] == 'COVID-19', 'disease_id'] = 'case'
adata.obs.loc[adata.obs['disease__ontology_label'] == 'long COVID-19', 'disease_id'] = 'case'
adata.obs.disease_id.value_counts()
adata.obs[['disease_id','donor_id']].value_counts().sort_index()
adata_full = adata.copy()


task_name = 'covid_cls'
sample_key = 'donor_id'
task_key = 'disease_id'
ct_key = 'celltype'
class_num = 2
folds = pd.read_csv(os.path.join(path, "covid_5folds.csv"), index_col=0)

save_dir = f"{RESULTS_DIR}/R2-q1/Results/proportion/Covid"
os.makedirs(save_dir, exist_ok=True)


def compute_metrics_macro(y_true_onehot, y_pred_proba, y_pred_logit):
    """auROC, macroAUPRC and macroF1 of one split from scikit-learn's macro averages.

    y_pred_logit is the 0/1 top-class indicator. AUROC and AUPRC are NaN when scikit-learn refuses them (e.g. a
    class missing from the split).
    """
    try:
        auroc = roc_auc_score(y_true_onehot, y_pred_proba)
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true_onehot, y_pred_proba, average="macro")
    except ValueError:
        auprc = float("nan")
    macro_f1 = f1_score(y_true_onehot, y_pred_logit, average="macro", zero_division=0)
    return {"auROC": auroc, "macroAUPRC": auprc, "macroF1": macro_f1}


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
        metrics_tr = compute_metrics_macro(y_train.values, y_pred_train, y_pred_train_logit.values)
        metrics_val = compute_metrics_macro(y_val.values, y_pred_val, y_pred_val_logit.values)
        metrics_te = compute_metrics_macro(
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
        metrics_tr = compute_metrics_macro(y_train.values, y_pred_train, y_pred_train_logit.values)
        metrics_val = compute_metrics_macro(y_val.values, y_pred_val, y_pred_val_logit.values)
        metrics_te = compute_metrics_macro(
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
