"""Stand-alone CloudPred run on Cardio (DCM / HCM / NF): donor bags, raw genes, 10 seeds x 5 folds.

Reads $DATA_DIR/cardio/cardio_processed.h5ad (obs donor_id, disease, cell_type_leiden0.6) and
$DATA_DIR/cardio/cardio_5folds.csv (label index, then 'samples' = donor and 'fold' 1-5), and imports cloudpred
from $CLOUDPRED_PATH (default ./CloudPred) or the installed package.

Each donor's expression matrix (genes x cells, sparse) is first written to ./data/cardio_cls/<disease>/<donor>.npz
with its cell types in ct_<donor>.npy (relative to the working directory) and read back; a donor's class index is
the position of its disease folder in sorted order (DCM, HCM, NF). Per run the test fold is held out and the
remaining donors are split 80/20 (train_test_split, random_state=SEED). cloudpred_train fits a 5-component
diagonal-covariance GaussianMixture on the pooled training cells, then trains the classifier's output layer by
full-batch SGD (momentum 0.9), SGD_STEPS_PER_LR steps for each rate in LEARNING_RATES, keeping the state with the
lowest validation loss. Donors are scored from the softmax of the outputs: macro AUROC and macro AUPRC
(per-class one-vs-rest AUROC and trapezoidal precision-recall area, averaged) and macro-F1 of the argmax class.

Writes to ./data/cardio_cls/: cardio_cls_results_fold_<fold>_seed_<seed>.csv (train / val / test metrics of one
run, one row per metric) and cardio_cls_results_all_folds.pth (all of them, keyed 'fold_<fold>_seed_<seed>').
$RESULTS_DIR/R2-q1/Results/CloudPred/Cardio/full_metrics/ is created but nothing is written there.
"""

# Paths come from environment variables (see experiments-public/README.md).
import os
DATA_DIR = os.environ.get("DATA_DIR", "data")  # input datasets
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")  # output root

import copy
import glob
import pathlib
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns
import sklearn.metrics
import sklearn.mixture
import torch
import torch.nn as nn
import tqdm
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from torch.utils.data import Dataset

# $CLOUDPRED_PATH (default ./CloudPred) goes first on sys.path, ahead of any installed cloudpred.
sys.path.insert(0, os.environ.get("CLOUDPRED_PATH", "CloudPred"))

import cloudpred
adata = sc.read_h5ad(
    f"{DATA_DIR}/cardio/cardio_processed.h5ad"
)
folds = pd.read_csv(
    f"{DATA_DIR}/cardio/cardio_5folds.csv",
    index_col=0,
)


num_folds = folds["fold"].nunique()

save_dir = (
    f"{RESULTS_DIR}/R2-q1/Results/CloudPred/Cardio/full_metrics"
)
os.makedirs(save_dir, exist_ok=True)

# Bare expressions left from interactive exploration; nothing below uses their values.
adata.obs.biosample_id.value_counts()
adata.obs.donor_id.value_counts()
adata.obs[['disease','donor_id']].value_counts().sort_index()
adata.obs['cell_type_leiden0.6'].value_counts()
adata_full = adata.copy()

class_num = 3
sample_key = 'donor_id'
task_key = 'disease'
task_name = 'cardio_cls'
ct_key = 'cell_type_leiden0.6'


data = adata.copy()
meta = adata.obs.copy()

# ./data/<task_name>/<class>/: one .npz (genes x cells) and one ct_*.npy file per donor
for state in set(data.obs[task_key]):
    pathlib.Path(os.path.join("data", task_name, state)).mkdir(parents=True, exist_ok=True)


for ind in tqdm.tqdm(set(data.obs[sample_key])):
        state = list(set(data.obs[data.obs[sample_key] == ind][task_key]))
        assert(len(state) == 1)
        state = state[0]
        X = data.X[(data.obs[sample_key] == ind).values, :].transpose()
        scipy.sparse.save_npz(os.path.join("data", task_name, state, "{}.npz".format(ind)), X)
        np.save(os.path.join("data", task_name, state, "ct_{}.npy".format(ind)),
                data.obs[data.obs[sample_key] == ind][ct_key].values)

import sklearn.metrics


def load_counts(filename):
    """One donor as (cells x genes sparse counts, cell types or None, donor id taken from the file name)."""
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


# Xall[k] holds (counts, k, cell types, donor id) for every donor in the k-th class folder (sorted order)
dir = './data/' + task_name
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
from sklearn.metrics import roc_auc_score, f1_score
from scipy.special import softmax
import numpy as np
import torch

def metric_output(X, model):
    """Macro AUROC, macro-F1 and macro AUPRC of model on the bags in X (tuples starting with features, label)."""
    y_true = []
    y_score = []

    for (_, (x, y, *_)) in enumerate(X):
        z = model(torch.Tensor(x))          # logits
        y_true.append(y)
        y_score.append(list(z[0].detach().numpy()))

    y_score = softmax(np.array(y_score), axis=1)

    y_true_oh = np.eye(class_num)[y_true]

    # One-vs-rest; classes whose column is constant in X are skipped
    auroc_vals = []
    for c in range(y_true_oh.shape[1]):
        if len(np.unique(y_true_oh[:, c])) > 1:
            auroc_vals.append(
                roc_auc_score(y_true_oh[:, c], y_score[:, c])
            )
    macro_auroc = np.mean(auroc_vals)

    macro_f1 = f1_score(
        np.array(y_true),
        y_score.argmax(1),
        average="macro"
    )

    # Trapezoidal precision-recall area, same class filter
    auprc_vals = []
    for c in range(y_true_oh.shape[1]):
        if len(np.unique(y_true_oh[:, c])) > 1:
            precision, recall, _ = precision_recall_curve(
                y_true_oh[:, c],
                y_score[:, c]
            )
            auprc_vals.append(auc(recall, precision))
    macro_auprc = np.mean(auprc_vals)

    return macro_auroc, macro_f1, macro_auprc


# Output-layer schedule: SGD_STEPS_PER_LR full-batch SGD steps at each rate, in this order
LEARNING_RATES = [1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]
SGD_STEPS_PER_LR = 1000


def cloudpred_train(Xtrain, Xvalid, centers=2, class_num=2):
    """CloudPred DensityClassifier with `centers` diagonal Gaussians fit on all training cells.

    Only the output layer (classifier.pl) is trained, by SGD at each rate in LEARNING_RATES; after each rate the
    state with the lowest Xvalid loss is restored.
    """
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
    X = torch.cat([mixture(torch.Tensor(X)).unsqueeze_(0).detach() for (X, y, *_) in Xtrain])
    y = torch.LongTensor([y for (X, y, *_) in Xtrain])
    Xv = torch.cat([mixture(torch.Tensor(X)).unsqueeze_(0).detach() for (X, y, *_) in Xvalid])
    yv = torch.LongTensor([y for (X, y, *_) in Xvalid])
    # best_loss restarts at every rate, so each rate ends on its own lowest-validation-loss state
    for lr in LEARNING_RATES:
        optimizer = torch.optim.SGD(classifier.pl.parameters(), lr=lr, momentum=0.9)
        criterion = torch.nn.modules.CrossEntropyLoss()
        best_loss = float("inf")
        best_model = copy.deepcopy(classifier.pl.state_dict())
        print("Learning rate: " + str(lr))
        for i in tqdm.tqdm(range(SGD_STEPS_PER_LR)):
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


project_pc = False
res_stat_dict = {}
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
        Xtrain, Xval = train_test_split(Xtrainval, test_size=0.2, random_state=SEED)
        Xtrain = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xtrain))
        Xval = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xval))
        Xtest  = list(map(lambda x: (np.array(x[0].todense()), *x[1:]), Xtest))


        import sklearn.mixture
        best_model = None
        best_score = float("inf")
        center = 5
        print(center)
        best_model = cloudpred_train(Xtrain, Xval, center, class_num=class_num)
        auROC_tr, macroF1_tr,macroAUPRC_tr = metric_output(Xtrain, model=best_model)
        auROC_val, macroF1_val,macroAUPRC_val = metric_output(Xval, model=best_model)
        auROC_te, macroF1_te,macroAUPRC_te = metric_output(Xtest, model=best_model)
        print(sample_key, '\n',
                            "auROC_tr", auROC_tr,'\n',
                            "auROC_val", auROC_val,'\n',
                            "macroF1_tr", macroF1_tr,'\n',
                            "macroF1_val", macroF1_val,'\n',
                            "auROC_test", auROC_te,'\n',
                            "macroF1_test", macroF1_te,'\n',
                            "fold", fold, '\n',
                            'seed', SEED
                            )
        res_df = pd.DataFrame({
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
        }, index=[0])
        res_df = res_df.T
        res_df.columns = [task_name]
        res_stat_dict['fold_' + str(fold) + '_seed_' + str(SEED)] = res_df
        res_df.to_csv(dir + '/' + task_name + '_results_fold_' + str(fold) + '_seed_' + str(SEED) +  '.csv')


# res_test_df (test metrics per run) is only built for inspection; it is not saved.
torch.save(res_stat_dict, dir + '/' + task_name + '_results_all_folds.pth')
res_stat_dict.keys()
res_test_df = pd.DataFrame(index=list(res_stat_dict.keys()),
                           columns=['auROC_test','macroF1_test','macroAUPRC_test'])
for i in list(res_stat_dict.keys()):
     print(i)
     res_test_df.loc[i,'auROC_test'] = res_stat_dict[i].loc['auROC_test'].values
     res_test_df.loc[i,'macroF1_test'] = res_stat_dict[i].loc['macroF1_test'].values
     res_test_df.loc[i,'macroAUPRC_test'] = res_stat_dict[i].loc['macroAUPRC_test'].values
