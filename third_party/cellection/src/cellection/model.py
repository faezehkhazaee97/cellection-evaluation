from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .utils import *
from .loss import *

import os
import copy
import random

import scipy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# Random Seed Setting: allow for reproducibility test
def set_seed(seed=None):
    if seed is None:
       seed = np.random.choice(int(1e2))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Parameter Initialization
def init_weights(nn_modules, method="xavier_normal"):
    if type(nn_modules) == nn.Linear:
        if method == "xavier_normal":
            torch.nn.init.xavier_normal_(nn_modules.weight)
        elif method == "kaiming_normal":
            torch.nn.init.kaiming_normal_(nn_modules.weight, mode='fan_in', nonlinearity='relu')
        elif method == "xavier_uniform":
            torch.nn.nn.init.xavier_uniform_(nn_modules.weight)
        elif method == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(nn_modules.weight, mode='fan_in', nonlinearity='relu')
        nn_modules.bias.data.fill_(0.0) if nn_modules.bias is not None else None


# Fully Connected Neural Networks: Linear -> LayerNorm -> Activation (-> BatchNorm) -> Dropout
def FCNN(layers, layernorm=True, activation=nn.ReLU(), batchnorm=False, dropout_rate=0):
    fc_nn = []
    for i in range(1, len(layers)):
        fc_nn.append(nn.Linear(layers[i-1], layers[i]))
        if layernorm:
            fc_nn.append(nn.LayerNorm(layers[i]))
        fc_nn.append(activation)
        if batchnorm:
            fc_nn.append(nn.BatchNorm1d(layers[i]))
        fc_nn.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*fc_nn)


# basic Conv1dBlock and FCBlock
class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, activation=nn.ReLU(), batch_norm=True):
        super(Conv1dBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation
        self.batch_norm = batch_norm
    def forward(self, x):   
        x = self.conv1d(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU(), batch_norm=False):
        super(FCBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = activation
        self.batch_norm = batch_norm
    def forward(self, x):
        x = self.linear(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.activation(x)
        return x


# ============================================================================
# T-net (Spatial Transformation Network)
class Tnet(nn.Module):
    def __init__(self, dim, conv1d_dims = [64, 256, 1024], fc_blocks=[512, 256]):
        super(Tnet, self).__init__()
        self.dim = dim 
        self.conv1d_dims = conv1d_dims
        self.fc_blocks = fc_blocks
        self.conv_num = len(conv1d_dims)
        self.fc_num = len(fc_blocks)
        prev_dim = dim
        for i in range(self.conv_num):
            setattr(self, f'conv{i+1}', Conv1dBlock(prev_dim, conv1d_dims[i], kernel_size=1))
            prev_dim = conv1d_dims[i]
        prev_dim = conv1d_dims[-1]
        for i in range(self.fc_num):
            setattr(self, f'fc{i+1}', FCBlock(prev_dim, fc_blocks[i]))
            prev_dim = fc_blocks[i]
        self.linear_final = nn.Linear(prev_dim, dim**2)
    def forward(self, x):
        batch_size = x.shape[0]
        num_points = x.shape[2]
        # Pass the input through each Conv1D block
        for i in range(self.conv_num):
            x = getattr(self, f'conv{i+1}')(x)
        # Pass the output through the MaxPool1D layer and reshape it
        x = F.max_pool1d(x, kernel_size=num_points).view(batch_size, -1)
        # Pass the output through each FC block
        for i in range(self.fc_num):
            x = getattr(self, f'fc{i+1}')(x)
        # Pass the output through the final Linear layer 
        x = self.linear_final(x)
        # Add an identity matrix to the output
        iden = torch.eye(self.dim, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.dim, self.dim) + iden
        return x


# ============================================================================
# each Input --> T-net --> Intermediate Output: 1 × 40 × 40 multiplied by 1 x 40 x cell_num = 1 x 40 x cell_num
# - the intermediate output is passed through the Shared MLP to scale up to 1 x 64 x cell_num
# - the output is passed through the T-net again to get the final output: 1 x 64 x cell_num
# - the output is passed through the Shared MLP to scale up to 1 x 128 x cell_num --> 1 x 1024 x cell_num 
# maxpool1d to get 1 x 1024 vector
# pass through the the ordinal classifier to get the final output (40 ordinal classes in total)
# ============================================================================
# Shared MLP (set it simple for now, can be more complex if needed)
    # The shared MLPs are implemented as 1D convolutions, 
    # since convolutions naturally allow for weight sharing they make shared MLPs easy to implement
class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SharedMLP, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.conv1 = Conv1dBlock(in_channels, mid_channels, kernel_size=1)
        self.conv2 = Conv1dBlock(mid_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x



# ===========================
# Aggregation Modules
# ===========================
class Aggregator(nn.Module):
    def __init__(self, method="gated_attention", input_dim=30, hidden_dim=64):
        super(Aggregator, self).__init__()
        self.method = method
        if method == "attention":
            self.attn = nn.Sequential(nn.Linear(input_dim, hidden_dim), 
                                      nn.Tanh(), 
                                      nn.Linear(hidden_dim, 1, bias=False)) # multiMIL , bias=False
        elif method == "gated_attention":
            self.attn_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
            self.attn_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
            self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, X):
        batch_size, num_feats, num_points = X.shape
        if self.method in ["max", "mean", "sum"]:
            if self.method == "max":
                global_features, self.critical_indexes = F.max_pool1d(X, kernel_size=num_points, return_indices=True)
                self.critical_indexes = self.critical_indexes.view(batch_size, -1)
            elif self.method == "mean":
                global_features = F.avg_pool1d(X, kernel_size=num_points)
                self.critical_indexes = None
            elif self.method == "sum":
                global_features = torch.sum(X, dim=2, keepdim=True)
                self.critical_indexes = None
            return global_features.view(batch_size, -1), self.critical_indexes
        elif self.method == "attention":
            self.A = self.attn(X.transpose(1, 2)).squeeze(-1)
            self.A = torch.softmax(self.A, dim=1).unsqueeze(1)
            return torch.sum(X * self.A, dim=2), self.A
            # # multiMIL implementation:
            # self.A = self.attn(X.transpose(1, 2))
            # self.A = torch.transpose(self.A, -1, -2)
            # self.A = F.softmax(self.A, dim=-1)
            # return torch.bmm(self.A, X.transpose(1, 2)).squeeze(dim=1)
        elif self.method == "gated_attention":
            A_V = self.attn_V(X.transpose(1, 2))
            A_U = self.attn_U(X.transpose(1, 2))
            self.A  = self.attn_weights(A_V * A_U).squeeze(-1)
            self.A = torch.softmax(self.A , dim=1).unsqueeze(1)
            return torch.sum(X * self.A , dim=2), self.A
            # # multiMIL implementation:
            # A_V = self.attn_V(X.transpose(1, 2))
            # A_U = self.attn_U(X.transpose(1, 2))
            # A = self.attn_weights(A_V * A_U)
            # A = torch.transpose(A, -1, -2)
            # A = F.softmax(A, dim=-1)
            # return torch.bmm(A, X.transpose(1, 2)).squeeze(dim=1)
        else:
            raise ValueError("Unknown aggregation method")


# ============================================================================
# Point Net Backbone (main Architecture)
class PointNetBackbone(nn.Module):
    def __init__(self, first_dim=40, second_dim=64, 
                 conv1d_dims=[64], 
                 fc_blocks=[256], 
                 global_features=1024,
                 batch_norm=True,
                 attention_dim = 64,
                 agg_method="attention",
                 use_tnet=False,
                 use_tnet1=None,
                 use_tnet2=None):  # Add aggregation method: max, mean, sum, attention, gated_attention
        super(PointNetBackbone, self).__init__()
        self.num_global_feats = global_features
        self.conv1d_dims = conv1d_dims + fc_blocks + [global_features]
        self.fc_blocks = [int(global_features / 2)] + fc_blocks
        # Spatial Transformer Networks (T-nets)
        self.use_tnet = use_tnet
        self.use_tnet1 = use_tnet if use_tnet1 is None else use_tnet1
        self.use_tnet2 = use_tnet if use_tnet2 is None else use_tnet2

        if self.use_tnet1:
            self.tnet1 = Tnet(dim=first_dim, conv1d_dims=self.conv1d_dims, fc_blocks=self.fc_blocks)
        else:
            self.tnet1 = None

        if self.use_tnet2:
            self.tnet2 = Tnet(dim=second_dim, conv1d_dims=self.conv1d_dims[1:], fc_blocks=self.fc_blocks)
        else:
            self.tnet2 = None

        # Shared MLP layers
        self.shared_mlp1 = SharedMLP(first_dim, 64)
        self.shared_mlp2 = SharedMLP(second_dim, global_features)
        self.bn_bool = batch_norm
        self.bn = nn.BatchNorm1d(self.num_global_feats)
        # Aggregator
        self.agg_method = agg_method
        self.aggregator = Aggregator(method=self.agg_method, input_dim=global_features, hidden_dim=attention_dim)
    def forward(self, x):
        batch_size, num_feats, num_points = x.shape # N x D x L
        # First transformation
        if self.use_tnet1:
            A_input = self.tnet1(x)
            x = torch.bmm(x.transpose(2,1), A_input).transpose(2,1)
        x = self.shared_mlp1(x)
        # Second transformation
        if self.use_tnet2:
            self.A_feat = self.tnet2(x)
            x = torch.bmm(x.transpose(2,1), self.A_feat).transpose(2,1)
        self.local_features = x.clone()
        x = self.shared_mlp2(x)
        if self.bn_bool:
            x = self.bn(x)
        # return x   # break point to test the intermediate output
        # Apply different aggregation methods
        if self.agg_method in ["max", "mean", "sum"]:
            global_features, critical_indexes = self.aggregator(x)
            return global_features, critical_indexes
        elif self.agg_method in ["attention", "gated_attention"]:
            global_features, attn_score = self.aggregator(x)
            return global_features, attn_score


# Encoder single cell
class CellEncoder(nn.Module):
    def __init__(self,
                    input_dim:int,
                    input_batch_num:int,
                    hidden_layer:int,
                    layernorm=True,
                    activation=nn.ReLU(),
                    batchnorm=False,
                    dropout_rate=0,
                    add_linear_layer=False,
                    clip_threshold=None):
        super(CellEncoder, self).__init__()
        self.add_linear_layer = add_linear_layer
        self.clip_threshold = clip_threshold
        # encoder latent representation for each cell
        self.hidden_rep = FCNN([input_dim+input_batch_num] + hidden_layer,
                                layernorm=layernorm,
                                activation=activation,
                                batchnorm=batchnorm,
                                dropout_rate=dropout_rate)
        # add an extra linear layer after hidden_z network to match the structure of encoder of AE/VAE
        if self.add_linear_layer is True:
            self.extra_linear_rep = nn.Linear(hidden_layer[-1], hidden_layer[-1], bias=True) # nn.Sequential(nn.Linear(hidden_layer[-1], hidden_layer[-1], bias=True), nn.LayerNorm(hidden_layer[-1]))
    def forward(self, input, input_batch):
        # infer the latent representation/ meta-feature
        if input_batch is not None:
            latent_rep = self.hidden_rep(torch.cat([input, input_batch], dim=1))
        else:
            latent_rep = self.hidden_rep(input)
        if self.add_linear_layer is True:
            latent_rep = self.extra_linear_rep(latent_rep)
        else:
            latent_rep = latent_rep
        return latent_rep



# ============================================================================
# Classification Head
class PointNetClassHead(nn.Module):
    def __init__(self, input_dim=60, conv1d_dims=[64], fc_blocks=[256], 
                 global_features=256, k=40, 
                 attention_dim = 64,
                 agg_method="attention",
                 use_tnet=False,
                 use_tnet1=None,
                 use_tnet2=None):  # Add aggregation method: max, mean, sum, attention, gated_attention
        super(PointNetClassHead, self).__init__()
        self.agg_method = agg_method  # Store aggregation method
        # Initialize the backbone with the chosen aggregation method
        self.backbone = PointNetBackbone(first_dim=input_dim, 
                                         conv1d_dims=conv1d_dims, 
                                         fc_blocks=fc_blocks, 
                                         global_features=global_features,
                                         attention_dim=attention_dim,
                                         agg_method=agg_method,
                                         use_tnet=use_tnet,
                                         use_tnet1=use_tnet1,
                                         use_tnet2=use_tnet2)
        # Classification MLP
        self.linear = nn.Linear(global_features, round(global_features/2))
        self.out = nn.Linear(round(global_features/2), k)
    def forward(self, x):
        # Forward pass through the backbone
        if self.agg_method in ["max", "mean", "sum"]:
            global_features, crit_idxs = self.backbone(x)
            attn_weights = None
        elif self.agg_method in ["attention", "gated_attention"]:
            global_features, attn_weights = self.backbone(x)
            crit_idxs = None
        # Check expected shape
        batch_size = x.shape[0]
        expected_shape = (batch_size, self.linear.in_features)  # Should match input of first Linear layer
        if global_features.shape != expected_shape:
            raise ValueError(f"Shape mismatch! Expected {expected_shape}, but got {global_features.shape}")
        # Pass through classification layers
        x = F.relu(self.linear(global_features))
        embedding = x.clone()
        output = self.out(x)
        self.global_features = global_features
        self.attn_weights = attn_weights
        self.crit_idxs = crit_idxs
        self.embedding = embedding
        return output, embedding, global_features, attn_weights, crit_idxs
