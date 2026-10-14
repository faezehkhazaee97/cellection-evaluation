from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .utils import *
from .loss import *
from .model import *

import os
import copy
import random
import tqdm

import scipy
import numpy as np
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns


def train_model(model):
    model.train()


def eval_model(model):
    model.eval()


class cellectiion_object():
    def __init__(self,
                adata,
                task_type,
                task_key,
                sample_key,
                batch_key,
                model_type = 'classification',
                input_type = 'measurement',
                sparse_input = True,
                InstanceEncoder = True,
                use_tnet=True,
                use_tnet1=None,
                use_tnet2=None,
                train_cutpoints=True,
                seed = 2,
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                hidden_layer = [256, 64],
                global_features = 256,
                attention_dim = 64,  
                aggregator = 'gated_attention',
                activation = nn.ReLU(),
                layernorm = True,
                batchnorm = False,
                dropout_rate = 0.1,
                learning_rate = 1e-4,
                batch_size = 15,
                val_size = 0.2,
                max_epochs = 200,
                patience = 20,
                prioritized_cell_pct = 0.1,
                attention_cutoff = 10,
                save_model = True,
                save_path = None):
        self.adata = adata
        self.task_type = task_type
        self.task_key = task_key
        self.sample_key = sample_key
        self.batch_key = batch_key
        self.model_type = model_type
        self.input_type = input_type
        self.sparse_input = sparse_input
        self.train_cutpoints=train_cutpoints
        self.InstanceEncoder = InstanceEncoder
        self.use_tnet = use_tnet
        self.use_tnet1 = use_tnet1
        self.use_tnet2 = use_tnet2
        self.val_size = val_size
        self.aggregator = aggregator
        self.global_features = global_features
        self.attention_dim = attention_dim
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        self.activation = activation
        self.hidden_layer = hidden_layer
        self.layernorm = layernorm
        self.batchnorm = batchnorm
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.prioritized_cell_pct = prioritized_cell_pct
        self.attention_cutoff = attention_cutoff
        self.save_model = save_model
        self.save_path = save_path
    def prepare(self):
        metadata = self.adata.obs.copy()
        if self.input_type == 'measurement':
            Ns = self.adata.X.copy()
            if self.sparse_input:
                Ns = Ns.todense()
            columns = self.adata.var.index.tolist()
            if len(columns) != Ns.shape[1]:
                columns = [f"gene_{i}" for i in range(Ns.shape[1])]
        else:
            Ns = self.adata.obsm[self.input_type].copy()
            columns = None
        Ns_df = pd.DataFrame(Ns, index=metadata.index.tolist(), columns=columns)
        if self.task_type == 'ordinal':
            Ys_df = self.adata.obs[[self.task_key]].copy()
            Ys_df[self.task_key] = Ys_df[self.task_key].astype(int)
        else:
            Ys_df = pd.get_dummies(self.adata.obs[self.task_key].copy()).astype(int)
        # LOCK label column order for this fold
        self.label_columns = Ys_df.columns.tolist()
        if self.batch_key is not None:
            batch_dummy = pd.get_dummies(self.adata.obs[self.batch_key].copy()).astype(int)
            Covs_df = pd.DataFrame(batch_dummy, index=metadata.index.tolist(), columns=batch_dummy.columns)
            Ns_df = pd.concat([Ns_df, Covs_df], axis=1)
        else:
            Covs_df = None
        Xs, Ys, ins, meta_ids = Create_MIL_Dataset(ins_df=Ns_df, label_df=Ys_df, metadata=metadata, bag_column=self.sample_key)
        print(f"Number of bags: {len(Xs)}", f"Number of labels: {Ys.shape}", f"Number of max instances: {max(ins)}")
        self.mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
        if self.task_type == 'classification':
            self.class_num = Ys.shape[1]
        elif self.task_type == 'regression':
            self.class_num = 1
        elif self.task_type == 'ordinal':
            self.num_ord_classes = int(Ys_df[self.task_key].max()) + 1  # = 8
            self.class_num = 1  # model outputs ONE scalar
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")
        self.train_size = int((1-self.val_size) * len(self.mil_dataset))
        self.val_size = len(self.mil_dataset) - self.train_size
        torch.manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(self.mil_dataset, [self.train_size, self.val_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=MIL_Collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.val_size, shuffle=False, collate_fn=MIL_Collate_fn)
        if self.InstanceEncoder:
            print("Input dimension for CellEncoder =", Ns_df.shape[1])

            self.enc = CellEncoder(input_dim=Ns_df.shape[1], input_batch_num=0, 
                                hidden_layer=self.hidden_layer, activation=self.activation,
                                layernorm=self.layernorm, batchnorm=self.batchnorm, dropout_rate=self.dropout_rate, 
                                add_linear_layer=True, clip_threshold=None)
            self.enc.apply(init_weights)
            self.enc.to(self.device)
            self.pn = PointNetClassHead(input_dim=self.hidden_layer[-1], k=self.class_num, 
                                        global_features=self.global_features, attention_dim=self.attention_dim, 
                                        agg_method=self.aggregator,
                                        use_tnet=self.use_tnet,
                                        use_tnet1=self.use_tnet1,
                                        use_tnet2=self.use_tnet2)
            self.pn.apply(init_weights)
            self.pn.to(self.device)
            self.optimizer = torch.optim.Adam(list(self.pn.parameters())+list(self.enc.parameters()), 
                                        lr=self.learning_rate)
        else:
            if Ns.shape[1] >= 3000:
                # give warning as the input dimension is too large
                print(f"Warning: The input dimension is too large for PointNet ({Ns_df.shape[1]}), please consider using InstanceEncoder.")
            self.enc = None
            self.pn = PointNetClassHead(input_dim=Ns_df.shape[1], k=self.class_num, 
                                            global_features=self.global_features, attention_dim=self.attention_dim, 
                                            agg_method=self.aggregator,
                                            use_tnet=self.use_tnet,
                                            use_tnet1=self.use_tnet1,
                                            use_tnet2=self.use_tnet2)
            self.pn.apply(init_weights)
            self.pn.to(self.device)
            self.optimizer = torch.optim.Adam(list(self.pn.parameters()), lr=self.learning_rate)
        if self.task_type == 'classification':
            self.criterion = nn.CrossEntropyLoss()
            self.transformation_function = nn.Softmax(dim=1)
        elif self.task_type == 'regression':
            self.criterion = nn.MSELoss()
        elif self.task_type == 'ordinal':
            self.criterion = OrdinalRegressionLoss(
                num_class=self.num_ord_classes,   # 8 → 7 cutpoints
                train_cutpoints=self.train_cutpoints
            )
            if self.train_cutpoints:
                # Ensure ordinal cutpoints are optimized when train_cutpoints=True.
                self.optimizer.add_param_group(
                    {
                        "params": self.criterion.parameters(),
                        "lr": self.learning_rate * 10.0,
                    }
                )
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")
    def train(self):
        if self.save_model:
            if self.save_path is None:
                self.save_path_temp = './' + self.task_key + '_' + self.model_type + '_SEED' + str(self.seed)
            else:
                self.save_path_temp = self.save_path + '/' + self.task_key + '_' + self.model_type + '_SEED' + str(self.seed)
            if not os.path.exists(self.save_path_temp):
                os.makedirs(self.save_path_temp)
        self.best_val_loss = float('inf')
        patience_counter = 0
        self.best_model_state = None
        if self.task_type == 'ordinal' and self.train_cutpoints:
            self.cutpoints_history = []
        for val_padded_bags, val_labels, val_lengths, val_id in self.val_loader:
            # print('val size', len(val_lengths))
            pass
        from tqdm import tqdm
        for epoch in tqdm(range(self.max_epochs)):
            for padded_bags, labels, lengths, _ in self.train_loader:
            

                self.optimizer.zero_grad()
                loss_tr = 0
                for idx in range(len(lengths)):
                    length = lengths[idx]
                    if length <= 1:
                        continue
                    else:
                        # padded_bags shape: (B, N, D)
                        input_tr = padded_bags[idx, :length, :]     # shape (N, D)

                        if self.InstanceEncoder:
                            # input_tr is (N_cells, N_genes) — CORRECT
                            enc_out = self.enc(input_tr, None)       # (N, hidden)
                            output_tr = self.pn(enc_out.T.unsqueeze(0))[0]
                        else:
                            input_tr_bn = input_tr.unsqueeze(0).permute(0, 2, 1)
                            output_tr = self.pn(input_tr_bn)[0]

                        if self.task_type == 'classification':
                            pred_label_tr = self.transformation_function(output_tr)
                            true_label_tr = labels[idx]
                            loss_per_sample = self.criterion(pred_label_tr, torch.max(true_label_tr.reshape(-1, self.class_num),1)[1])
                            loss_tr += loss_per_sample
                        elif self.task_type == 'regression':
                            pred_label_tr = output_tr
                            true_label_tr = labels[idx]
                            loss_per_sample = self.criterion(pred_label_tr, true_label_tr)
                            loss_tr += loss_per_sample
                        elif self.task_type == 'ordinal':
                            loss_per_sample = self.criterion(output_tr.to('cpu'), labels[idx].view(-1, 1).to('cpu')) 
                            loss_tr += loss_per_sample.to(self.device)
                (loss_tr/len(lengths)).backward()
                self.optimizer.step()
            print(f"Epoch [{epoch+1}/{self.max_epochs}], Train Loss: {loss_tr:.4f}")
            
            loss_val = 0
            
            for val_idx in range(len(val_lengths)):
                val_length = val_lengths[val_idx]
                if val_length <= 1:
                    continue

                # Correct shape: (1, D, N) because padded_bags already permuted earlier
                input_val = val_padded_bags[val_idx, :val_length, :]   # (N, D)

                if self.InstanceEncoder:
                    enc_val = self.enc(input_val, None)                 # (N, hidden)
                    output_val = self.pn(enc_val.T.unsqueeze(0))[0]
                else:
                    input_val_bn = input_val.unsqueeze(0).permute(0, 2, 1)
                    output_val = self.pn(input_val_bn)[0]


                # Compute loss
                if self.task_type == 'classification':
                    pred_label_val = self.transformation_function(output_val)
                    true_label_val = val_labels[val_idx]
                    loss_per_sample = self.criterion(
                        pred_label_val, torch.max(true_label_val.reshape(-1, self.class_num), 1)[1]
                    )
                    loss_val += loss_per_sample
                elif self.task_type == 'regression':
                    pred_label_val = output_val
                    true_label_val = val_labels[val_idx]
                    loss_per_sample = self.criterion(pred_label_val, true_label_val)
                    loss_val += loss_per_sample
                elif self.task_type == 'ordinal':
                    loss_per_sample = self.criterion(
                        output_val.to("cpu"), val_labels[val_idx].view(-1, 1).to("cpu")
                    )
                    loss_val += loss_per_sample.to(self.device)

            loss_val_avg = loss_val/len(val_lengths)
            print(f"Epoch [{epoch+1}/{self.max_epochs}], Val Loss: {loss_val_avg:.4f}")
            if self.task_type == 'ordinal' and self.train_cutpoints:
                self.cutpoints_history.append(
                    self.criterion.cutpoints.detach().cpu().numpy().copy()
                )
            if loss_val_avg < self.best_val_loss:
                self.best_val_loss = loss_val_avg
                if self.save_model:
                    torch.save(self.pn, self.save_path_temp + "/best_pn_" + self.task_key + '_' + str(self.seed) + ".pt") 
                    torch.save(self.enc, self.save_path_temp + "/best_enc_" + self.task_key + '_' + str(self.seed) + ".pt") if self.InstanceEncoder else None
                patience_counter = 0
                print(f"Saving the best model with validation loss: {self.best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        if self.task_type == 'ordinal' and self.train_cutpoints and self.save_model:
            cutpoints_df = pd.DataFrame(self.cutpoints_history)
            cutpoints_df.to_csv(
                self.save_path_temp + "/cutpoints_history.csv",
                index=False
            )
        if self.save_model:
            self.pn_checkpoint = torch.load(self.save_path_temp + "/best_pn_" + self.task_key + '_' + str(self.seed) + ".pt") 
            self.enc_checkpoint = torch.load(self.save_path_temp + "/best_enc_" + self.task_key + '_' + str(self.seed) + ".pt") if self.InstanceEncoder else None
        else:
            # Model is not saved, please check the save_model and save_path
            raise ValueError("Model is not saved, please check the save_model and save_path")
    def inference(self):
        inference_data_loader = DataLoader(self.mil_dataset, batch_size=len(self.mil_dataset), 
                                            shuffle=False, collate_fn=MIL_Collate_fn_CPU_Inference)
        for all_padded_bags, all_labels, all_lengths, all_id in inference_data_loader:
            pass
        pred_label_list = []
        true_label_list = []
        instance_level_list = []
        attention_mtx_list = []
        cell_id_list = []
        embedding_list = []
        global_feature_list = []
        self.pn_checkpoint.to('cpu')
        self.enc_checkpoint.to('cpu') if self.InstanceEncoder else None
        for all_idx in range(len(all_lengths)):
            all_length = all_lengths[all_idx]
            if all_length <= 1:
                continue
            else:
                input_all = all_padded_bags[all_idx, :all_length, :]   # (N, D)

                if self.InstanceEncoder:
                    enc_all = self.enc_checkpoint(input_all, None)
                    output_all = self.pn_checkpoint(enc_all.T.unsqueeze(0))
                else:
                    input_all_bn = input_all.unsqueeze(0).permute(0, 2, 1)
                    output_all = self.pn_checkpoint(input_all_bn)

                output_y, output_emb, output_gf, output_attn, _ = output_all
                attention_mtx_list.append(output_attn.squeeze(0,1).detach().cpu().numpy())
                cell_id_list.append(all_id[all_idx])
                embedding_list.append(output_emb.squeeze(0).detach().cpu().numpy())
                global_feature_list.append(output_gf.squeeze(0).detach().cpu().numpy())
                true_label_all = all_labels[all_idx]
                true_label_list.append(true_label_all.detach().cpu().numpy())
                if self.task_type == 'classification':
                    pred_label_all = self.transformation_function(output_y)
                    pred_label_list.append(pred_label_all.detach().cpu().numpy())
                elif self.task_type == 'regression':
                    pred_label_all = output_y.clone()
                    pred_label_list.append(pred_label_all.detach().cpu().numpy())
                elif self.task_type == 'ordinal':
                    pred_label_all = output_y.clone()
                    pred_label_list.append(pred_label_all.detach().cpu().numpy())
        if self.task_type == 'classification':
            true_label_all_df = pd.DataFrame(
                np.vstack(true_label_list),
                columns=self.label_columns
            )

            pred_label_all_df = pd.DataFrame(
                np.vstack(pred_label_list),
                columns=self.label_columns
            )
        elif self.task_type == 'regression':
            true_label_all_df = pd.DataFrame(np.vstack(true_label_list), columns=[self.task_key])
            pred_label_all_df = pd.DataFrame(np.vstack(pred_label_list), columns=[self.task_key])
        elif self.task_type == 'ordinal':
            true_label_all_df = pd.DataFrame(np.vstack(true_label_list), columns=[self.task_key])
            pred_label_all_df = pd.DataFrame(np.vstack(pred_label_list), columns=[self.task_key])
        sample_meta = [i[0] for i in cell_id_list]
        cell_id_list = [i for sublist in cell_id_list for i in sublist]
        cell_id_df = pd.DataFrame([i for i in cell_id_list], columns=["cell_id"])
        metadata_all = self.adata.obs.loc[cell_id_df.cell_id]
        sample_meta_all = metadata_all.loc[sample_meta]
        true_label_all_df.index = sample_meta_all[self.sample_key]
        pred_label_all_df.index = sample_meta_all[self.sample_key]
        embedding_df = pd.DataFrame(np.vstack(embedding_list))
        global_feature_df = pd.DataFrame(np.vstack(global_feature_list))
        embedding_df.index = sample_meta_all[self.sample_key]
        global_feature_df.index = sample_meta_all[self.sample_key]
        attention_mtx_raw = np.concatenate(attention_mtx_list, axis=0) 
        attention_mtx_raw_df = pd.DataFrame(attention_mtx_raw, columns=["attention_score_raw"], index=cell_id_df.cell_id)
        attention_mtx_list_norm_cellnum = [i*len(i) for i in attention_mtx_list]
        attention_mtx_norm_cellnum = np.concatenate(attention_mtx_list_norm_cellnum, axis=0)
        attention_mtx_raw_df['attention_score_norm_cellnum'] = attention_mtx_norm_cellnum.tolist()
        metadata_all = metadata_all.join(attention_mtx_raw_df)
        self.adata.obs = metadata_all.loc[self.adata.obs.index].copy()
        sample_meta_all.index = sample_meta_all[self.sample_key]
        sample_meta_all = sample_meta_all[[self.sample_key, self.task_key, self.batch_key]] if self.batch_key is not None else sample_meta_all[[self.sample_key, self.task_key]]
        return sample_meta_all, true_label_all_df, pred_label_all_df, embedding_df, global_feature_df, attention_mtx_raw_df
    def prepare_inference(self):
        """
        Build a MIL dataset and dataloader for inference only (no train/val split).
        """
        metadata = self.adata.obs.copy()
        if self.input_type == 'measurement':
            Ns = self.adata.X.copy()
            if self.sparse_input:
                Ns = Ns.todense()
            columns = self.adata.var.index.tolist()
        else:
            # Use embeddings from .obsm
            Ns = self.adata.obsm[self.input_type].copy()
            columns = None
        Ns_df = pd.DataFrame(
            Ns,
            index=metadata.index.tolist(),
            columns=columns
        )
        if self.task_type == 'ordinal':
            # Ordinal labels must be integer ranks
            Ys_df = self.adata.obs[[self.task_key]].copy()
            Ys_df[self.task_key] = Ys_df[self.task_key].astype(int)
        else:
            # Classification / regression
            Ys_df = pd.get_dummies(
                self.adata.obs[self.task_key].copy()
            ).astype(int)
            Ys_df = Ys_df.reindex(columns=self.label_columns, fill_value=0)

        if self.batch_key is not None:
            batch_dummy = pd.get_dummies(
                self.adata.obs[self.batch_key].copy()
            ).astype(int)
            Covs_df = pd.DataFrame(
                batch_dummy,
                index=metadata.index.tolist(),
                columns=batch_dummy.columns
            )
            Ns_df = pd.concat([Ns_df, Covs_df], axis=1)
        Xs, Ys, ins, meta_ids = Create_MIL_Dataset(
            ins_df=Ns_df,
            label_df=Ys_df,
            metadata=metadata,
            bag_column=self.sample_key
        )
        print(f"[Inference] Number of bags: {len(Xs)}, Max instances: {max(ins)}")
        self.mil_dataset = MILDataset(Xs, Ys, ins, meta_ids)
        self.mil_dataset.meta_ids = meta_ids
        self.inference_loader = DataLoader(
            self.mil_dataset,
            batch_size=len(self.mil_dataset),   # load all bags at once
            shuffle=False,
            collate_fn=MIL_Collate_fn_CPU_Inference
        )
