"""Asian vs White donors of the lupus cohort ("Ethnicity" in Fig. 2 / Table S3), ProtoCell4P with 8
prototypes.

lupus.py with obs 'pop_cov' (ASIAN / WHITE) as the label and its own fold file,
$DATA_DIR/lupus/race_5folds.csv. Everything else, the binary auROC included, is as in lupus.py.
Results land in $RESULTS_DIR/R2-q1/Results/4_ProtoCell4P/Lupus_Race/proto_8/ (task lupus_race_cls).
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
import tqdm
from sklearn.metrics import auc, precision_recall_curve
from torch.utils.data import Dataset


def _to_numpy(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def compute_macro_auprc_from_logits(y_true, y_logits):
    """Macro AUPRC from integer labels and per-class scores (NaN if no class has both label values)."""
    y_true = _to_numpy(y_true).astype(int)
    y_logits = _to_numpy(y_logits)
    num_classes = y_logits.shape[1]
    y_true_onehot = np.zeros((len(y_true), num_classes), dtype=int)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    scores = []
    for c in range(num_classes):
        if len(np.unique(y_true_onehot[:, c])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, c], y_logits[:, c])
        scores.append(auc(recall, precision))
    return float(np.mean(scores)) if scores else float("nan")


class ProtoCell(nn.Module):
    """Prototype-based MIL classifier (ProtoCell4P) for bags of cells.

    Encoder, mirrored decoder and importance network each have n_layers hidden layers of width h_dim. A
    cell's class logits are its prototype similarities 1 / (squared distance + 0.5) weighted by its
    sigmoid importance scores; a bag's logits are the mean over its cells. lambdas holds the loss weights
    lambda_1 ... lambda_6, n_ct=None disables the cell-type heads, and prototype pairs closer than d_min
    are penalised.
    """

    def __init__(self, input_dim, h_dim, z_dim, n_layers, n_proto, n_classes, lambdas, n_ct=None, device="cpu", d_min=1):
        super(ProtoCell, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_proto = n_proto
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_ct = n_ct
        self.d_min = d_min
        assert self.n_layers > 0
        self.lambda_1 = lambdas["lambda_1"]
        self.lambda_2 = lambdas["lambda_2"]
        self.lambda_3 = lambdas["lambda_3"]
        self.lambda_4 = lambdas["lambda_4"]
        self.lambda_5 = lambdas["lambda_5"]
        self.lambda_6 = lambdas["lambda_6"]
        self.enc_i = nn.Linear(self.input_dim, self.h_dim)
        self.enc_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.enc_z = nn.Linear(self.h_dim, self.z_dim)
        self.dec_z = nn.Linear(self.z_dim, self.h_dim)
        self.dec_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.dec_i = nn.Linear(self.h_dim, self.input_dim)
        self.imp_i = nn.Linear(self.input_dim, self.h_dim)
        self.imp_h = nn.Sequential(*([nn.Linear(self.h_dim, self.h_dim) for _ in range(self.n_layers - 1)]))
        self.imp_p = nn.Linear(self.h_dim, self.n_proto * self.n_classes)
        self.activate = nn.LeakyReLU()
        self.prototypes = nn.parameter.Parameter(torch.empty(self.n_proto, self.z_dim), requires_grad = True)
        self.ce_ = nn.CrossEntropyLoss(reduction="mean")
        nn.init.xavier_normal_(self.enc_i.weight)
        nn.init.xavier_normal_(self.enc_z.weight)
        nn.init.xavier_normal_(self.dec_z.weight)
        nn.init.xavier_normal_(self.dec_i.weight)
        nn.init.xavier_normal_(self.imp_i.weight)
        nn.init.xavier_normal_(self.imp_p.weight)
        nn.init.xavier_normal_(self.prototypes)
        for i in range(self.n_layers - 1):
            nn.init.xavier_normal_(self.enc_h[i].weight)
            nn.init.xavier_normal_(self.dec_h[i].weight)
            nn.init.xavier_normal_(self.imp_h[i].weight)
        if self.n_ct is not None:
            self.ct_clf1 = nn.Linear(self.n_proto, self.n_ct)
            self.ct_clf2 = nn.Linear(self.n_proto * self.n_classes, self.n_ct)
            nn.init.xavier_normal_(self.ct_clf1.weight)
            nn.init.xavier_normal_(self.ct_clf2.weight)

    def forward(self, x, y, ct=None, sparse=True):
        """Bag classification; returns (loss, bag logits[, per-cell cell-type logits when ct is given]).

        x is a list of cells x genes matrices (scipy sparse if sparse=True), ct per-bag lists of cell-type
        ids. loss = bag cross-entropy + lambda_6 * cell-type cross-entropy from the importance scores.
        """
        # All bags' cells are stacked; split_idx keeps each bag's row range for the per-bag mean.
        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])
        if sparse:
            x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
        else:
            x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)
        z = self.encode(x)
        import_scores = self.compute_importance(x) # (n_cell, n_proto, n_class)
        c2p_dists = torch.pow(z[:, None] - self.prototypes[None, :], 2).sum(-1)
        c_logits = (1 / (c2p_dists+0.5))[:,None,:].matmul(import_scores).squeeze(1) # (n_cell, n_classes)
        logits = torch.stack([c_logits[split_idx[i]:split_idx[i+1]].mean(dim=0) for i in range(len(split_idx)-1)])
        clf_loss = self.ce_(logits, y)
        if self.n_ct is not None and ct is not None:
            ct_logits = self.ct_clf2(import_scores.reshape(-1, self.n_proto * self.n_classes))
            ct_loss = self.ce_(ct_logits, torch.tensor([j for i in ct for j in i]).to(self.device))
        else:
            ct_loss = 0
        total_loss = clf_loss + self.lambda_6 * ct_loss
        if ct is not None:
            return total_loss, logits, ct_logits
        return total_loss, logits

    def encode(self, x):
        """Cells x genes -> cells x z_dim embedding."""
        h_e = self.activate(self.enc_i(x))
        for i in range(self.n_layers - 1):
            h_e = self.activate(self.enc_h[i](h_e))
        z = self.activate(self.enc_z(h_e))
        return z

    def decode(self, z):
        """Embedding -> reconstructed expression (non-negative)."""
        h_d = self.activate(self.dec_z(z))
        for i in range(self.n_layers - 1):
            h_d = self.activate(self.dec_h[i](h_d))
        x_hat = torch.relu(self.dec_i(h_d))
        return x_hat

    def compute_importance(self, x):
        """Per-cell prototype importance for each class, shape (n_cells, n_proto, n_classes), in [0, 1]."""
        h_i = self.activate(self.imp_i(x))
        for i in range(self.n_layers - 1):
            h_i = self.activate(self.imp_h[i](h_i))
        import_scores = torch.sigmoid(self.imp_p(h_i)).reshape(-1, self.n_proto, self.n_classes)
        return import_scores

    def pretrain(self, x, y, ct=None, sparse=True):
        """Representation losses, same inputs as forward; returns (loss, cell-type logits) if ct is given.

        Weighted by lambda_1 ... lambda_5: reconstruction MSE; each cell's squared distance to its nearest
        prototype; for each prototype, the smallest mean squared distance to the cells of one cell type,
        with a random 30% of prototype / cell-type pairs masked; a squared hinge on prototype pairs closer
        than d_min; cell-type cross-entropy from the prototype similarities.
        """
        split_idx = [0]
        for i in range(len(x)):
            split_idx.append(split_idx[-1]+x[i].shape[0])
        if sparse:
            x = torch.cat([torch.tensor(x[i].toarray()) for i in range(len(x))]).to(self.device)
        else:
            x = torch.cat([torch.tensor(x[i]) for i in range(len(x))]).to(self.device)
        y = y.to(self.device)
        if ct is not None:
            ct = torch.tensor([j for i in ct for j in i]).to(self.device)
        z = self.encode(x)
        x_hat = self.decode(z)
        c2p_dists = torch.pow(z[:, None] - self.prototypes[None, :], 2).sum(-1)
        if ct is None:
            p2c_dists = torch.pow(self.prototypes[:, None] - z[None, :], 2).sum(-1)
        else:
            p2c_dists = torch.stack([torch.pow(self.prototypes[:, None] - z[ct == t][None, :], 2).sum(-1).mean(-1) for t in ct.unique().tolist()]).T # n_proto * n_ct
        p2p_dists = (torch.pow(self.prototypes[:, None] - self.prototypes[None, :], 2).sum(-1)+1e-16).sqrt()
        recon_loss = (x - x_hat).pow(2).mean()
        c2p_loss = (c2p_dists).min(dim=1)[0].mean()
        p2c_loss = (p2c_dists + (torch.ones_like(p2c_dists).uniform_() < 0.3) * 1e9).min(dim=1)[0].mean()
        p2p_loss = ((self.d_min - p2p_dists > 0) * (self.d_min - p2p_dists)).pow(2).sum() / (self.n_proto * self.n_proto - self.n_proto)
        if self.n_ct is not None and ct is not None:
            ct_logits = self.ct_clf1(1 / (c2p_dists+0.5))
            ct_loss = self.ce_(ct_logits, ct)
        else:
            ct_loss = 0
        total_loss = self.lambda_1 * recon_loss +\
                     self.lambda_2 * c2p_loss +\
                     self.lambda_3 * p2c_loss +\
                     self.lambda_4 * p2p_loss +\
                     self.lambda_5 * ct_loss
        if ct is not None:
            return total_loss, ct_logits
        return total_loss


class OurDataset(Dataset):
    """Items are (X[i], y[i], ct[i]) when ct_id is given, else (X[i], y[i]); X[i] is sparse cells x genes.

    cell_id, gene_id and class_id are only stored.
    """

    def __init__(self, X, y, cell_id=None, gene_id=None, class_id=None, ct=None, ct_id=None):
        self.X = X
        self.y = y
        self.cell_id = cell_id
        self.gene_id = gene_id
        self.class_id = class_id
        self.ct = ct
        self.ct_id = ct_id

    def __getitem__(self, i):
        if self.ct_id is not None:
            return self.X[i], self.y[i], self.ct[i]
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)


path = f"{DATA_DIR}/lupus/"
adata = sc.read_h5ad(os.path.join(path, "lupus_processed.h5ad"))
# Label and bag counts (bare expressions; visible only in an interactive session).
adata.obs.disease_cov.value_counts()
adata.obs.pop_cov.value_counts()
adata_full = adata.copy()
adata.obs.ind_cov.value_counts()
adata.obs[['pop_cov','ind_cov']].value_counts().sort_index()

task_name = 'lupus_race_cls'
sample_key = 'ind_cov'
task_key = 'pop_cov'
ct_key = 'ct_cov'
class_num = 2
folds = pd.read_csv(os.path.join(path, "race_5folds.csv"), index_col=0)

# Fig. 2 compares 4, 8 and 16 prototypes; each value gets its own proto_<n> folder.
n_proto = 8
save_dir = f"{RESULTS_DIR}/R2-q1/Results/4_ProtoCell4P/Lupus_Race/proto_{n_proto}"
os.makedirs(save_dir, exist_ok=True)


res_stat_dict = {}
metrics_rows = []
for SEED in range(10):
    for fold in [1,2,3,4,5]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folds_trainval = folds[folds['fold'] != fold]
        folds_test = folds[folds['fold'] == fold]
        adata = adata_full[adata_full.obs[sample_key].isin(folds_trainval['samples'])]
        adata_test = adata_full[adata_full.obs[sample_key].isin(folds_test['samples'])]
        barcodes = adata.obs.index.tolist()
        genes = adata.var.index.tolist()
        meta = adata.obs.copy()
        cell_types = meta[ct_key]
        ct_id = sorted(set(cell_types))
        mapping_ct = {c:idx for idx, c in enumerate(ct_id)}
        X = []
        y = []
        ct = []
        # Per bag: sparse cells x genes, label, per-cell cell-type ids. Labels become ints in sorted
        # class-name order.
        for ind in tqdm.tqdm(sorted(set(meta[sample_key]))):
                disease = list(set(meta[task_key][meta[sample_key] == ind]))
                mask = (meta[sample_key] == ind).to_numpy()
                x = adata.X[mask]
                X.append(x)
                y.append(disease[0])
                ct.append([mapping_ct[c] for c in cell_types[mask]])
        class_id = sorted(set(y))
        mapping = {c:idx for idx, c in enumerate(class_id)}
        y = [mapping[c] for c in y]
        len(y)
        dataset = OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id, ct=ct, ct_id=ct_id)
        from sklearn.model_selection import train_test_split
        train_set, val_set = train_test_split(dataset, test_size=0.2, random_state=SEED)
        torch.manual_seed(SEED)  # seeds model init and DataLoader shuffling
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # lambda_5 / lambda_6 weight the cell-type losses and drop to 0 without cell types.
        lr = 1e-4
        max_epoch = 20
        batch_size = 4
        test_step = 1
        h_dim = 128
        z_dim = 32
        d_min = 1
        n_layers = 2
        n_classes = max(dataset.y) + 1
        if dataset.ct is None:
            n_ct = None
        else:
            n_ct = len(dataset.ct_id)
        ct_id = dataset.ct_id
        class_id = dataset.class_id
        pretrained=False
        max_epoch_pretrain = 0
        lr_pretrain = 1e-2
        lambda_1,lambda_2, lambda_3, lambda_4, lambda_5, lambda_6 = 1,1,1,1,1,1
        lambdas = {
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "lambda_3": lambda_3,
                    "lambda_4": lambda_4,
                    "lambda_5": 0 if n_ct is None else lambda_5,
                    "lambda_6": 0 if n_ct is None else lambda_6
                }
        model_type = "ProtoCell"
        model = ProtoCell(dataset.X[0].shape[1], h_dim, z_dim, n_layers, n_proto, n_classes, lambdas, n_ct, device, d_min)
        model.to(device)
        def my_collate(batch):
                """No padding: x and ct stay per-bag lists, y becomes a tensor."""
                x = [item[0] for item in batch]
                y = torch.tensor([item[1] for item in batch])
                if len(batch[0]) == 3:
                    ct = [item[2] for item in batch]
                    return x, y, ct
                return x, y
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, collate_fn=my_collate)
        val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = False, collate_fn=my_collate)
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        best_epoch = 0
        best_metric = 0
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        for epoch in range(max_epoch):
            train_loss = 0
            y_truth = []
            y_logits = []
            if n_ct is not None:
                ct_truth = []
                ct_logits = []
            for bat in train_loader:
                x = bat[0]
                y = bat[1]
                optim.zero_grad()
                if n_ct is not None:
                    loss, logits, ct_logit = model(*bat, sparse=True)
                else:
                    loss, logits = model(*bat, sparse=True)
                # pretrained=False: the pretrain() losses join every training step.
                if not pretrained:
                    if n_ct is not None:
                        loss += model.pretrain(*bat, sparse=True)[0]
                    else:
                        loss += model.pretrain(*bat, sparse=True)
                loss.backward()
                optim.step()
                train_loss += loss.item() * len(x)
                y_truth.append(y)
                y_logits.append(torch.softmax(logits, dim=1))
                if n_ct is not None:
                    ct_truth.append(torch.tensor([j for i in bat[2] for j in i]))
                    ct_logits.append(torch.softmax(ct_logit, dim=1))
            y_truth = torch.cat(y_truth)
            y_logits = torch.cat(y_logits)
            y_pred = y_logits.argmax(dim=1)
            train_acc = accuracy_score(y_truth.cpu(), y_pred.cpu())
            if y_logits.shape[1] == 2:
                train_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach()[:,1])
            else:
                train_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach(), multi_class="ovo")
            train_f1 = f1_score(y_truth.cpu(), y_pred.cpu(), average="macro")
            if n_ct is not None:
                ct_truth = torch.cat(ct_truth)
                ct_logits = torch.cat(ct_logits)
                ct_pred = ct_logits.argmax(dim=1)
                ct_acc = accuracy_score(ct_truth.cpu(), ct_pred.cpu())
                print("Avg. Training Loss: {:.2f} | Avg. Training Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f} | CT Acc: {:.2f}".format(train_loss / len(train_set), train_acc, train_auc, train_f1, ct_acc))
            else:
                print("Avg. Training Loss: {:.2f} | Avg. Training Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f}".format(train_loss / len(train_set), train_acc, train_auc, train_f1))
            # Checkpoint criterion: val_auc + ct_acc / 10.
            if (epoch + 1) % test_step == 0:
                model.eval()
                val_loss = 0
                y_truth = []
                y_logits = []
                if n_ct is not None:
                    ct_truth = []
                    ct_logits = []
                with torch.no_grad():
                    for bat in val_loader:
                        x = bat[0]
                        y = bat[1]
                        if n_ct is not None:
                            loss, logits, ct_logit = model(*bat, sparse=True)
                        else:
                            loss, logits = model(*bat, sparse=True)
                        val_loss += loss.item() * len(x)
                        y_truth.append(y)
                        y_logits.append(torch.softmax(logits, dim=1))
                        if n_ct is not None:
                            ct_truth.append(torch.tensor([j for i in bat[2] for j in i]))
                            ct_logits.append(torch.softmax(ct_logit, dim=1))
                    y_truth = torch.cat(y_truth)
                    y_logits = torch.cat(y_logits)
                    y_pred = y_logits.argmax(dim=1)
                    val_acc = accuracy_score(y_truth.cpu(), y_pred.cpu())
                    if y_logits.shape[1] == 2:
                        val_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach()[:,1])
                    else:
                        val_auc = roc_auc_score(y_truth.cpu(), y_logits.cpu().detach(), multi_class="ovo")
                    val_f1 = f1_score(y_truth.cpu(), y_pred.cpu(), average="macro")
                if n_ct is not None:
                    ct_truth = torch.cat(ct_truth)
                    ct_logits = torch.cat(ct_logits)
                    ct_pred = ct_logits.argmax(dim=1)
                    ct_acc = accuracy_score(ct_truth.cpu(), ct_pred.cpu())
                    curr_metric = val_auc + ct_acc / 10
                    print("Avg. Validation Loss: {:.2f} | Avg. Validation Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f} | CT Acc: {:.2f}".format(val_loss / len(val_set), val_acc, val_auc, val_f1, ct_acc))
                else:
                    curr_metric = val_auc
                    print("Avg. Validation Loss: {:.2f} | Avg. Validation Accuracy: {:.2f} | Avg. ROC AUC Score: {:.2f} | F1 Score: {:.2f}".format(val_loss / len(val_set), val_acc, val_auc, val_f1))
                if curr_metric > best_metric:
                    torch.save(model, os.path.join(save_dir, f"best_model_{task_name}_{fold}_seed_{SEED}.pt"))
                    best_metric = curr_metric
                    best_epoch = epoch
                    print("Model Saved!")
                model.train()
        # Best checkpoint; each split is scored as a single batch.
        model = torch.load(os.path.join(save_dir, f"best_model_{task_name}_{fold}_seed_{SEED}.pt"))
        model.eval()
        train_loader_ = DataLoader(train_set, batch_size = len(train_set), shuffle = True, collate_fn=my_collate)
        val_loader_ = DataLoader(val_set, batch_size = len(val_set), shuffle = False, collate_fn=my_collate)
        for bat in train_loader_:
            x = bat[0]
            y = bat[1]
            if n_ct is not None:
                loss, logits, ct_logit = model(*bat, sparse=True)
            else:
                loss, logits = model(*bat, sparse=True)
            y_truth_tr = y.clone().cpu()
            y_logits_tr = torch.softmax(logits, dim=1).detach().cpu()
            y_pred_tr = y_logits_tr.argmax(dim=1)
        # Two classes: the second line overwrites the probability AUROC with the AUROC of the argmax
        # labels, and that is the value stored. Same for val and test.
        if y_logits_tr.shape[1] == 2:
            auROC_tr = roc_auc_score(y_truth_tr, y_logits_tr[:,1])
            auROC_tr = roc_auc_score(y_truth_tr, y_pred_tr)
        else:
            auROC_tr = roc_auc_score(y_truth_tr, y_logits_tr, multi_class="ovo")
        macroF1_tr = f1_score(y_truth_tr, y_pred_tr, average="macro")
        macroAUPRC_tr = compute_macro_auprc_from_logits(y_truth_tr, y_logits_tr)
        for bat in val_loader_:
            x = bat[0]
            y = bat[1]
            if n_ct is not None:
                loss, logits, ct_logit = model(*bat, sparse=True)
            else:
                loss, logits = model(*bat, sparse=True)
            y_truth_val = y.clone().cpu()
            y_logits_val= torch.softmax(logits, dim=1).detach().cpu()
            y_pred_val = y_logits_val.argmax(dim=1)
        if y_logits_val.shape[1] == 2:
            auROC_val = roc_auc_score(y_truth_val, y_logits_val[:,1])
            auROC_val = roc_auc_score(y_truth_val, y_pred_val)
        else:
            auROC_val = roc_auc_score(y_truth_val, y_logits_val, multi_class="ovo")
        macroF1_val = f1_score(y_truth_val, y_pred_val, average="macro")
        macroAUPRC_val = compute_macro_auprc_from_logits(y_truth_val, y_logits_val)
        # Test bags, built as above. Class ids come from the test fold's own sorted names, so they match
        # training only if every class occurs in the fold.
        barcodes = adata_test.obs.index.tolist()
        genes = adata_test.var.index.tolist()
        meta = adata_test.obs.copy()
        cell_types = meta[ct_key]
        ct_id = sorted(set(cell_types))
        mapping_ct = {c:idx for idx, c in enumerate(ct_id)}
        X = []
        y = []
        ct = []
        for ind in tqdm.tqdm(sorted(set(meta[sample_key]))):
                disease = list(set(meta[task_key][meta[sample_key] == ind]))
                mask = (meta[sample_key] == ind).to_numpy()
                x = adata_test.X[mask]
                X.append(x)
                y.append(disease[0])
                ct.append([mapping_ct[c] for c in cell_types[mask]])
        class_id = sorted(set(y))
        mapping = {c:idx for idx, c in enumerate(class_id)}
        y = [mapping[c] for c in y]
        len(y)
        test_set = OurDataset(X=X, y=y, cell_id=barcodes, gene_id=genes, class_id=class_id, ct=ct, ct_id=ct_id)
        test_loader = DataLoader(test_set, batch_size = len(test_set), shuffle = True, collate_fn=my_collate)
        for bat in test_loader:
            x = bat[0]
            y = bat[1]
            if n_ct is not None:
                loss, logits, ct_logit = model(*bat, sparse=True)
            else:
                loss, logits = model(*bat, sparse=True)
            y_truth_te = y.clone().cpu()
            y_logits_te = torch.softmax(logits, dim=1).detach().cpu()
            y_pred_te = y_logits_te .argmax(dim=1)
        if y_logits_te.shape[1] == 2:
            auROC_te = roc_auc_score(y_truth_te, y_logits_te[:,1])
            auROC_te = roc_auc_score(y_truth_te, y_pred_te)
        else:
            auROC_te = roc_auc_score(y_truth_te, y_logits_te, multi_class="ovo")
        macroF1_te = f1_score(y_truth_te, y_pred_te, average="macro")
        macroAUPRC_te = compute_macro_auprc_from_logits(y_truth_te, y_logits_te)
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
                    "model": "ProtoCell4P",
                    "seed": SEED,
                    "fold": fold,
                    "split": split_name,
                    "auROC": metrics[0],
                    "macroF1": metrics[1],
                    "macroAUPRC": metrics[2],
                }
            )


torch.save(res_stat_dict, os.path.join(save_dir, f"{task_name}_results_all_folds.pth"))


# Per-run test table; its .mean() is an interactive leftover and prints nothing here.
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
metrics_df.to_csv(os.path.join(save_dir, f"{task_name}_metrics_all.csv"), index=False)
