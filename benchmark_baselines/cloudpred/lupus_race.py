"""Ethnicity task of Fig. 2 / Table S3 with CloudPred: Asian vs White donors of the lupus cohort.

lupus.py with obs 'pop_cov' (ASIAN / WHITE) as the label and $DATA_DIR/lupus/race_5folds.csv as the fold
file. Outputs: $RESULTS_DIR/R2-q1/Results/cloudpred/LupusRace/, task name lupus_race_cls.
"""

# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import copy
import glob
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns
import sklearn.mixture
import torch
import torch.nn as nn
import tqdm
from scipy.special import softmax
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import Dataset


path = f'{DATA_DIR}/lupus/'
adata = sc.read_h5ad(path + 'lupus_processed.h5ad')
# Quick look at labels and bags (bare expressions, shown only when run interactively).
adata.obs.disease_cov.value_counts()
adata.obs.pop_cov.value_counts()

save_dir = f"{RESULTS_DIR}/R2-q1/Results/cloudpred/LupusRace"
os.makedirs(save_dir, exist_ok=True)

adata_full = adata.copy()
adata.obs.ind_cov.value_counts()
adata.obs[['pop_cov','ind_cov']].value_counts().sort_index()

task_name = 'lupus_race_cls'
sample_key = 'ind_cov'
task_key = 'pop_cov'
ct_key = 'ct_cov'
class_num = 2
folds = pd.read_csv(path + 'race_5folds.csv', index_col=0)


# Export each bag in CloudPred's layout: data/<task_name>/<label>/<bag>.npz (genes x cells) and
# ct_<bag>.npy.
data = adata.copy()
meta = adata.obs.copy()
data_dir = os.path.join(save_dir, "data", task_name)
for state in set(data.obs[task_key]):
    pathlib.Path(os.path.join(data_dir, state)).mkdir(parents=True, exist_ok=True)


for ind in tqdm.tqdm(set(data.obs[sample_key])):
        state = list(set(data.obs[data.obs[sample_key] == ind][task_key]))
        assert(len(state) == 1)
        state = state[0]
        X = data.X[(data.obs[sample_key] == ind).values, :].transpose()
        scipy.sparse.save_npz(os.path.join(data_dir, state, "{}.npz".format(ind)), X)
        np.save(os.path.join(data_dir, state, "ct_{}.npy".format(ind)),
                data.obs[data.obs[sample_key] == ind][ct_key].values)


def load_counts(filename):
    """One exported bag as (cells x genes float matrix, cell types or None, bag id from the file name)."""
    counts = scipy.sparse.load_npz(filename)
    counts = counts.astype(float)
    ct_filename = os.path.join(os.path.dirname(filename), "ct_" + os.path.splitext(os.path.basename(filename))[0] + ".npy")
    if os.path.isfile(ct_filename):
        ct = np.load(ct_filename, allow_pickle=True)
    else:
        ct = None
    return counts.transpose(), ct, filename.split('/')[-1].split('.')[0]


def scipy_sparse_to_pytorch(x):
    x = x.tocoo()
    v = torch.Tensor(x.data)
    i = torch.LongTensor([x.row, x.col])
    return torch.sparse.FloatTensor(i, v, x.shape)


# Xall[k] holds the bags of the k-th label folder in sorted order, each as (cells x genes, k,
# cell types, bag id).
dir = data_dir
Xall = []
state = []
sample_id_all = []
for dirname in sorted(glob.iglob(dir + "/*")):
    X = []
    for filename in tqdm.tqdm(sorted(glob.iglob(dirname + "/*.npz"))):
        X.append(load_counts(filename))
    X = list(map(lambda x: (x[0], len(state), x[1], x[2]), X))
    state.append(os.path.basename(dirname))
    Xall.append(X)


import cloudpred


def compute_macro_auprc(y_true_onehot, y_score):
    """Mean PR-curve area over one-hot class columns, skipping single-valued ones (NaN if none)."""
    scores = []
    for c in range(y_true_onehot.shape[1]):
        if len(np.unique(y_true_onehot[:, c])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, c], y_score[:, c])
        scores.append(auc(recall, precision))
    return float(np.mean(scores)) if scores else float("nan")


def metric_output(X, model):
    """(auROC, macroF1, macroAUPRC) of `model` on a list of bags, from the softmax of its outputs.

    auROC averages one-vs-rest AUROCs over the classes present; class_num comes from the module.
    """
    y_true = []
    y_score = []
    for (start, (x, y, *_)) in enumerate(X):
        z = model(torch.Tensor(x))
        y_true.append(y)
        y_score.append(list(z[0].detach().numpy()))
    y_score = softmax(np.array(y_score), axis=1)
    class_num
    y_true = np.eye(class_num)[y_true]
    auroc = np.mean([
            roc_auc_score(y_true[:, c], y_score[:, c])
            for c in range(y_true.shape[1])
            if len(np.unique(y_true[:, c])) > 1
            ])
    macro_f1 = f1_score(y_true.argmax(1), y_score.argmax(1), average="macro")
    macro_auprc = compute_macro_auprc(y_true, y_score)
    return auroc, macro_f1, macro_auprc


# Classifier schedule: STEPS_PER_LR full-batch SGD steps at each rate, in this order.
CLASSIFIER_LRS = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]
STEPS_PER_LR = 1000


def cloudpred_train(Xtrain, Xvalid, centers=2, class_num=2):
    """Fits the GMM on all training cells, then trains only classifier.pl; returns the DensityClassifier."""
    X = np.concatenate([x for (x, *_) in Xtrain])
    gm = []
    for X, *_ in Xtrain:
        gm.append(X)
    gm = np.concatenate(gm)
    model = sklearn.mixture.GaussianMixture(n_components = centers, covariance_type='diag')
    gm = model.fit(gm)
    component = [cloudpred.cloudpred.Gaussian(torch.Tensor(gm.means_[i, :]),
                          torch.Tensor(1. / gm.covariances_[i, :])) for i in range(centers)]
    mixture = cloudpred.cloudpred.Mixture(component, gm.weights_)
    classifier = cloudpred.cloudpred.DensityClassifier(mixture, centers, class_num)
    # The mixture stays fixed, so each bag is summarised once here and only classifier.pl is optimised.
    X = torch.cat([mixture(torch.Tensor(X)).unsqueeze_(0).detach() for (X, y, *_) in Xtrain])
    y = torch.LongTensor([y for (X, y, *_) in Xtrain])
    Xv = torch.cat([mixture(torch.Tensor(X)).unsqueeze_(0).detach() for (X, y, *_) in Xvalid])
    yv = torch.LongTensor([y for (X, y, *_) in Xvalid])
    # Each stage starts from the previous stage's best state, but best_loss is reset, so a later stage
    # can end on a higher validation loss than an earlier one.
    for lr in CLASSIFIER_LRS:
        optimizer = torch.optim.SGD(classifier.pl.parameters(), lr=lr, momentum=0.9)
        criterion = torch.nn.modules.CrossEntropyLoss()
        best_loss = float("inf")
        best_model = copy.deepcopy(classifier.pl.state_dict())
        print("Learning rate: " + str(lr))
        for i in tqdm.tqdm(range(STEPS_PER_LR)):
            z = classifier.pl(X)
            loss = criterion(z, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            zv = classifier.pl(Xv)
            loss = criterion(zv, yv)
            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(classifier.pl.state_dict())
        classifier.pl.load_state_dict(best_model)
    return classifier


# Pass 1: raw genes (model 'CloudPred', the numbers in Fig. 2).
project_pc = False
res_stat_dict = {}
metrics_rows = []
for SEED in range(10):
    for fold in [1,2,3,4,5]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds_trainval = folds[folds['fold'] != fold]
        folds_test = folds[folds['fold'] == fold]
        # x[-1] is the bag id; flatten the per-label lists.
        Xtest = [[x for x in class_group if x[-1] in folds_test.samples.tolist()] for class_group in Xall]
        Xtest = [x for class_group in Xtest for x in class_group]
        Xtrainval = [[x for x in class_group if x[-1] in folds_trainval.samples.tolist()] for class_group in Xall]
        Xtrainval = [x for class_group in Xtrainval for x in class_group]
        from sklearn.model_selection import train_test_split
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        Xtrain, Xval = train_test_split(Xtrainval, test_size=0.2, random_state=SEED)
        if project_pc == True:
            dims = 50
            iterations = 25
            transform = 'none'
            figroot = None
            pc = cloudpred.utils.train_pca_autoencoder(Xtrain = scipy.sparse.vstack(list(map(lambda x: x[0], Xtrain))), # training cells x genes
                                                    Ytrain = None,
                                                    Xtest = scipy.sparse.vstack(list(map(lambda x: x[0], Xval))),    # validation cells x genes
                                                    Ytest = None,
                                                    dims = dims, transform = transform, iterations=iterations)
            pc = pc[:, :dims]
            # Project onto the components, centred on the training-cell mean.
            mu = scipy.sparse.vstack(list(map(lambda x: x[0], Xtrain))).mean(axis=0)            # 1 x genes
            Xtrain = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtrain))
            Xval = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xval))
            Xtest  = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtest))
            # Re-centre, then divide by one scalar: the std of the first component (sigma[0, 0]).
            full = np.concatenate(list(map(lambda x: x[0], Xtrain))) # training cells x 50
            mu = np.mean(full, axis=0)                               # 1 x 50
            sigma = np.sqrt(np.mean(np.square(full - mu), axis=0))   # 1 x 50
            sigma = sigma[0, 0]
            Xtrain = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xtrain))
            Xval = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xval))
            Xtest  = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xtest))
        else:
            Xtrain = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xtrain))
            Xval = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xval))
            Xtest  = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xtest))
        import sklearn.mixture
        best_model = None
        best_score = float("inf")
        center = 5
        print(center)
        best_model = cloudpred_train(Xtrain, Xval, center, class_num=class_num)
        auROC_tr, macroF1_tr, macroAUPRC_tr = metric_output(Xtrain, model=best_model)
        auROC_val, macroF1_val, macroAUPRC_val = metric_output(Xval, model=best_model)
        auROC_te, macroF1_te, macroAUPRC_te = metric_output(Xtest, model=best_model)
        print(sample_key, '\n',
                            "auROC_tr", auROC_tr,'\n',
                            "auROC_val", auROC_val,'\n',
                            "macroF1_tr", macroF1_tr,'\n',
                            "macroF1_val", macroF1_val,'\n',
                            "macroAUPRC_tr", macroAUPRC_tr,'\n',
                            "macroAUPRC_val", macroAUPRC_val,'\n',
                            "auROC_test", auROC_te,'\n',
                            "macroF1_test", macroF1_te,'\n',
                            "macroAUPRC_test", macroAUPRC_te,'\n',
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
                    'auROC_test': auROC_te,
                    'macroF1_test': macroF1_te,
                    'macroAUPRC_test': macroAUPRC_te,
                    'fold': fold
                }, index=[0]
            )
        res_df = res_df.T
        res_df.columns = [task_name]
        res_stat_dict['fold_' + str(fold) + '_seed_' + str(SEED)] = res_df
        res_df.to_csv(os.path.join(save_dir, f"{task_name}_results_fold_{fold}_seed_{SEED}.csv"))
        # One row per split for <task>_metrics_all.csv.
        for split_name, metrics in [
            ("train", (auROC_tr, macroF1_tr, macroAUPRC_tr)),
            ("val", (auROC_val, macroF1_val, macroAUPRC_val)),
            ("test", (auROC_te, macroF1_te, macroAUPRC_te)),
        ]:
            metrics_rows.append(
                {
                    "model": "CloudPred",
                    "seed": SEED,
                    "fold": fold,
                    "split": split_name,
                    "auROC": metrics[0],
                    "macroF1": metrics[1],
                    "macroAUPRC": metrics[2],
                }
            )


torch.save(res_stat_dict, os.path.join(save_dir, f"{task_name}_results_all_folds.pth"))
res_stat_dict.keys()
res_test_df = pd.DataFrame(index=list(res_stat_dict.keys()),
                           columns=['auROC_test','macroF1_test','macroAUPRC_test'])
for i in list(res_stat_dict.keys()):
     print(i)
     res_test_df.loc[i,'auROC_test'] = res_stat_dict[i].loc['auROC_test'].values
     res_test_df.loc[i,'macroF1_test'] = res_stat_dict[i].loc['macroF1_test'].values
     res_test_df.loc[i,'macroAUPRC_test'] = res_stat_dict[i].loc['macroAUPRC_test'].values


# Pass 2: PCA-autoencoder features (model 'CloudPred_PC', files prefixed PC_).
project_pc = True
res_stat_dict_pc = {}
for SEED in range(10):
    for fold in [1,2,3,4,5]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds_trainval = folds[folds['fold'] != fold]
        folds_test = folds[folds['fold'] == fold]
        Xtest = [[x for x in class_group if x[-1] in folds_test.samples.tolist()] for class_group in Xall]
        Xtest = [x for class_group in Xtest for x in class_group]
        Xtrainval = [[x for x in class_group if x[-1] in folds_trainval.samples.tolist()] for class_group in Xall]
        Xtrainval = [x for class_group in Xtrainval for x in class_group]
        from sklearn.model_selection import train_test_split
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        Xtrain, Xval = train_test_split(Xtrainval, test_size=0.2, random_state=SEED)
        if project_pc == True:
            dims = 50
            iterations = 25
            transform = 'none'
            figroot = None
            pc = cloudpred.utils.train_pca_autoencoder(Xtrain = scipy.sparse.vstack(list(map(lambda x: x[0], Xtrain))), # training cells x genes
                                                    Ytrain = None,
                                                    Xtest = scipy.sparse.vstack(list(map(lambda x: x[0], Xval))),    # validation cells x genes
                                                    Ytest = None,
                                                    dims = dims, transform = transform, iterations=iterations)
            pc = pc[:, :dims]
            mu = scipy.sparse.vstack(list(map(lambda x: x[0], Xtrain))).mean(axis=0)            # 1 x genes
            Xtrain = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtrain))
            Xval = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xval))
            Xtest  = list(map(lambda x: (x[0].dot(pc) - np.matmul(mu, pc), *x[1:]), Xtest))
            full = np.concatenate(list(map(lambda x: x[0], Xtrain))) # training cells x 50
            mu = np.mean(full, axis=0)                               # 1 x 50
            sigma = np.sqrt(np.mean(np.square(full - mu), axis=0))   # 1 x 50
            sigma = sigma[0, 0]
            Xtrain = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xtrain))
            Xval = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xval))
            Xtest  = list(map(lambda x: (np.array((x[0] - mu) / sigma), *x[1:]), Xtest))
        else:
            Xtrain = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xtrain))
            Xval = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xval))
            Xtest  = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xtest))
        import sklearn.mixture
        best_model = None
        best_score = float("inf")
        center = 5
        print(center)
        best_model = cloudpred_train(Xtrain, Xval, center, class_num=class_num)
        auROC_tr, macroF1_tr, macroAUPRC_tr = metric_output(Xtrain, model=best_model)
        auROC_val, macroF1_val, macroAUPRC_val = metric_output(Xval, model=best_model)
        auROC_te, macroF1_te, macroAUPRC_te = metric_output(Xtest, model=best_model)
        print(sample_key, '\n',
                            "auROC_tr", auROC_tr,'\n',
                            "auROC_val", auROC_val,'\n',
                            "macroF1_tr", macroF1_tr,'\n',
                            "macroF1_val", macroF1_val,'\n',
                            "macroAUPRC_tr", macroAUPRC_tr,'\n',
                            "macroAUPRC_val", macroAUPRC_val,'\n',
                            "auROC_test", auROC_te,'\n',
                            "macroF1_test", macroF1_te,'\n',
                            "macroAUPRC_test", macroAUPRC_te,'\n',
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
                    'auROC_test': auROC_te,
                    'macroF1_test': macroF1_te,
                    'macroAUPRC_test': macroAUPRC_te,
                    'fold': fold
                }, index=[0]
            )
        res_df = res_df.T
        res_df.columns = [task_name]
        res_stat_dict_pc['fold_' + str(fold) + '_seed_' + str(SEED)] = res_df
        res_df.to_csv(os.path.join(save_dir, f"PC_{task_name}_results_fold_{fold}_seed_{SEED}.csv"))
        for split_name, metrics in [
            ("train", (auROC_tr, macroF1_tr, macroAUPRC_tr)),
            ("val", (auROC_val, macroF1_val, macroAUPRC_val)),
            ("test", (auROC_te, macroF1_te, macroAUPRC_te)),
        ]:
            metrics_rows.append(
                {
                    "model": "CloudPred_PC",
                    "seed": SEED,
                    "fold": fold,
                    "split": split_name,
                    "auROC": metrics[0],
                    "macroF1": metrics[1],
                    "macroAUPRC": metrics[2],
                }
            )


torch.save(res_stat_dict_pc, os.path.join(save_dir, f"PC_{task_name}_results_all_folds.pth"))
res_stat_dict_pc.keys()
res_test_df_pc = pd.DataFrame(index=list(res_stat_dict_pc.keys()),
                           columns=['auROC_test','macroF1_test','macroAUPRC_test'])
for i in list(res_stat_dict_pc.keys()):
     print(i)
     res_test_df_pc.loc[i,'auROC_test'] = res_stat_dict_pc[i].loc['auROC_test'].values
     res_test_df_pc.loc[i,'macroF1_test'] = res_stat_dict_pc[i].loc['macroF1_test'].values
     res_test_df_pc.loc[i,'macroAUPRC_test'] = res_stat_dict_pc[i].loc['macroAUPRC_test'].values


res_test_df.to_csv(os.path.join(save_dir, f"{task_name}_results_all_folds.csv"))
res_test_df_pc.to_csv(os.path.join(save_dir, f"PC_{task_name}_results_all_folds.csv"))

# Both passes; read by summarize.py.
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(os.path.join(save_dir, f"{task_name}_metrics_all.csv"), index=False)


# Interactive leftovers; nothing is printed when run as a script.
task_name
res_test_df.mean()

res_test_df_pc.mean()
