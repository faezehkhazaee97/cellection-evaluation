# bioPointNet_Apr2025.py: the CELLECTION model code imported by celection/*.py (the Figure 2 runs) and by
# oligodendrocyte_subtypes/train_biopointnet.py. Copied unchanged from the authors' development code
# (bioPointNet_dev/bioPointNet_Apr2025.py, last modified 2025-10-31); these first lines are the only addition.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import copy

# ===========================
# Creating MIL Dataset from Ns_df, Ys_df, and metadata
# ===========================
def Create_MIL_Dataset(ins_df, label_df, metadata, bag_column):
    """
    Create a Multiple Instance Learning (MIL) dataset from a dataframe of instances, labels, and metadata.
    Args:
        ins_df (pd.DataFrame): Dataframe of instances with shape (num_instances, num_features).
        label_df (pd.DataFrame): Dataframe of labels with shape (num_instances, num_classes).
        metadata (pd.DataFrame): Dataframe of metadata with shape (num_instances, num_metadata).
        bag_column (str): Column name in metadata to group instances into bags.
    Returns:
        Xs (list): List of bags, where each bag is a numpy array of shape (num_instances, num_features).
        Ys (np.array): Array of labels for each bag, shape (num_bags, num_classes).
        ins (list): Number of instances in each bag.
        metadata_indices (list): List of metadata indices for each bag.
    """
    bags = {}
    for idx, row in metadata.iterrows():
        bag_id = row[bag_column]
        if bag_id not in bags:
            bags[bag_id] = []
        bags[bag_id].append(idx)
    Xs, Ys, ins, metadata_indices = [], [], [], []
    for bag_id, indices in bags.items():
        Xs.append(ins_df.loc[indices].values.astype(np.float32))
        Ys.append(label_df.loc[indices].values[0])  # Assuming label is the same for all instances in a bag
        ins.append(len(indices))  # Number of instances in each bag
        metadata_indices.append(indices)  # Save metadata indices for each bag
    return Xs, np.array(Ys), ins, metadata_indices


class MILDataset(Dataset):
    def __init__(self, Xs, Ys, ins, metadata_indices):
        """
        Args:
            Xs (list): List of bags, where each bag is a numpy array of shape (num_instances, num_features).
            Ys (np.array): Array of labels for each bag, shape (num_bags, num_classes).
            ins (list): Number of instances in each bag.
            metadata_indices (list): List of metadata indices for each bag.
        """
        self.Xs = Xs
        self.Ys = Ys
        self.ins = ins
        self.metadata_indices = metadata_indices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __len__(self):
        return len(self.Xs)
    def __getitem__(self, idx):
        """
        Returns:
            bag (torch.Tensor): Tensor of shape (num_instances, num_features).
            label (torch.Tensor): Tensor of shape (num_classes,).
            ins_ct (int): Number of instances in the bag.
            metadata_idx (list): List of metadata indices for the bag.
        """
        bag = torch.FloatTensor(self.Xs[idx]).to(self.device)  # Convert bag to tensor and move to device
        label = torch.tensor(self.Ys[idx], dtype=torch.long).to(self.device)  # Convert label to tensor and move to device
        ins_ct = self.ins[idx]  # Number of instances in the bag
        metadata_idx = self.metadata_indices[idx]  # Metadata indices for the bag
        return bag, label, ins_ct, metadata_idx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def MIL_Collate_fn(batch):
    """
    Custom collate function for MIL datasets: pads bags to the same length.
    Args:
        batch (list): List of tuples (bag, label) from the dataset.
    Returns:
        padded_bags (torch.Tensor): Padded bags of shape (batch_size, max_instances, num_features).
        labels (torch.Tensor): Labels of shape (batch_size, num_classes).
        lengths (torch.Tensor): Lengths of each bag in the batch, shape (batch_size,).
    """
    bags, labels, ins, metadata_idx= zip(*batch)
    lengths = ins
    max_length = max(ins)                            # Maximum number of instances in the batch
    # Pad bags to the same length
    padded_bags = torch.zeros((len(bags), max_length, bags[0].shape[1]), dtype=torch.float32)
    for i, bag in enumerate(bags):
        padded_bags[i, :len(bag)] = bag
    labels = torch.stack(labels)        # Stack labels into a single tensor
    return padded_bags.to(device), labels.to(device), lengths, metadata_idx


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
# ============================================================================
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
                 agg_method="attention"):  # Add aggregation method: max, mean, sum, attention, gated_attention
        super(PointNetBackbone, self).__init__()
        self.num_global_feats = global_features
        self.conv1d_dims = conv1d_dims + fc_blocks + [global_features]
        self.fc_blocks = [int(global_features / 2)] + fc_blocks
        # Spatial Transformer Networks (T-nets)
        self.tnet1 = Tnet(dim=first_dim, conv1d_dims=self.conv1d_dims, fc_blocks=self.fc_blocks)
        self.tnet2 = Tnet(dim=second_dim, conv1d_dims=self.conv1d_dims[1:], fc_blocks=self.fc_blocks)
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
        A_input = self.tnet1(x)
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
        x = self.shared_mlp1(x)
        # Second transformation
        self.A_feat = self.tnet2(x)
        x = torch.bmm(x.transpose(2, 1), self.A_feat).transpose(2, 1)
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


class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_class, train_cutpoints=False, scale=20.0):
        super().__init__()
        self.num_classes = num_class
        num_cutpoints = self.num_classes - 1
        self.cutpoints = torch.arange(num_cutpoints).float()*scale/(num_class-2) - scale / 2
        self.cutpoints = nn.Parameter(self.cutpoints)
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)
    def forward(self, pred, label):
        sigmoids = torch.sigmoid(self.cutpoints - pred)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )
        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)
        neg_log_likelihood = torch.log(likelihoods)
        if label is None:
            loss = 0
        else:
            loss = -torch.gather(neg_log_likelihood, 1, label).mean()  
        return loss#, likelihoods


# ============================================================================
# Classification Head
class PointNetClassHead(nn.Module):
    def __init__(self, input_dim=60, conv1d_dims=[64], fc_blocks=[256], 
                 global_features=256, k=40, 
                 attention_dim = 64,
                 agg_method="attention"):  # Add aggregation method: max, mean, sum, attention, gated_attention
        super(PointNetClassHead, self).__init__()
        self.agg_method = agg_method  # Store aggregation method
        # Initialize the backbone with the chosen aggregation method
        self.backbone = PointNetBackbone(first_dim=input_dim, 
                                         conv1d_dims=conv1d_dims, 
                                         fc_blocks=fc_blocks, 
                                         global_features=global_features,
                                         attention_dim=attention_dim,
                                         agg_method=agg_method)
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


# ============================================================================
# Classification Head
class PointNetClassHead(nn.Module):
    def __init__(self, input_dim=60, conv1d_dims=[64], fc_blocks=[256], 
                 global_features=256, k=40, 
                 attention_dim = 64,
                 agg_method="attention"):  # Add aggregation method: max, mean, sum, attention, gated_attention
        super(PointNetClassHead, self).__init__()
        self.agg_method = agg_method  # Store aggregation method
        # Initialize the backbone with the chosen aggregation method
        self.backbone = PointNetBackbone(first_dim=input_dim, 
                                         conv1d_dims=conv1d_dims, 
                                         fc_blocks=fc_blocks, 
                                         global_features=global_features,
                                         attention_dim=attention_dim,
                                         agg_method=agg_method)
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


# ============================================================================
# Multi-Task Classification Head
class PointNetClass_MultiHead(nn.Module):
    def __init__(self, input_dim=None, conv1d_dims=[64], fc_blocks=[256], 
                 global_features=256, task_class_nums=[2], 
                 attention_dim=64, agg_methods=["gated_attention"]):
        """
        Multi-task classification head for PointNet.
        Args:
            input_dim (int): Input feature dimension.
            conv1d_dims (list): Dimensions for 1D convolutional layers.
            fc_blocks (list): Dimensions for fully connected layers.
            global_features (int): Dimension of global features.
            task_class_nums (list): List of class numbers for each task. 
            [A value of 1 indicates regression or ordinal regression, etc.]
            attention_dim (int): Dimension for attention mechanism.
            agg_methods (list): List of aggregation methods for each task.
        """
        super(PointNetClass_MultiHead, self).__init__()
        self.task_class_nums = task_class_nums
        self.num_tasks = len(task_class_nums)
        self.agg_methods = agg_methods
        # Backbone to extract global features
        self.backbone = PointNetBackbone(first_dim=input_dim, 
                                         conv1d_dims=conv1d_dims, 
                                         fc_blocks=fc_blocks, 
                                         global_features=global_features,
                                         attention_dim=attention_dim,
                                         agg_method="none")  # No aggregation in backbone
        # Create aggregation and classification heads for each task
        self.aggregators = nn.ModuleList()
        self.classification_heads = nn.ModuleList()
        for i, num_classes in enumerate(task_class_nums):
            # Aggregation head
            self.aggregators.append(Aggregator(method=agg_methods[i], input_dim=global_features, hidden_dim=attention_dim))
            # Classification head
            if num_classes == 1:
                # Regression or ordinal regression task
                self.classification_heads.append(nn.Linear(global_features, 1))
            else:
                # Classification task
                self.classification_heads.append(nn.Sequential(
                    nn.Linear(global_features, round(global_features / 2)),
                    nn.ReLU(),
                    nn.Linear(round(global_features / 2), num_classes)
                ))
    def forward(self, x):
        """
        Forward pass for multi-task classification.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, input_dim).
        Returns:
            list: List of outputs for each task.
        """
        # Pass through backbone to get global features
        global_features = self.backbone(x)  # Shape: (batch_size, global_features)
        # Process each task
        task_outputs = []
        for i in range(self.num_tasks):
            # Aggregation
            agg_features, attn_weights = self.aggregators[i](global_features.unsqueeze(-1))
            # Classification
            task_output = self.classification_heads[i](agg_features)
            task_outputs.append(task_output)
        return task_outputs


class New_Aggregator(nn.Module):
    def __init__(self, method, input_dim, hidden_dim=64, 
                num_heads=4):
        """
        Aggregator module with support for self-attention and multi-head attention.
        Args:
            method (str): Aggregation method. Options: 
                        ["max", "mean", "sum", "attention", "gated_attention", "self_attention", "multihead_attention"].
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension for attention mechanisms.
            num_heads (int): Number of attention heads for multi-head attention ONLY.
        """
        super(New_Aggregator, self).__init__()
        self.method = method
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        if method == "attention":
            self.attn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), 
                nn.Tanh(), 
                nn.Linear(hidden_dim, 1, bias=False)  # Multi-MIL, bias=False
            )
        elif method == "gated_attention":
            self.attn_V = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), 
                nn.Tanh()
            )
            self.attn_U = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), 
                nn.Sigmoid()
            )
            self.attn_S = nn.Linear(hidden_dim, 1, bias=False)
        elif method == "self_attention":
            # Custom self-attention mechanism
            self.Query = nn.Linear(input_dim, hidden_dim)
            self.Key = nn.Linear(input_dim, hidden_dim)
            self.Value = nn.Linear(input_dim, hidden_dim)
            self.proj = nn.Linear(hidden_dim, input_dim)
            self.layer_norm = nn.LayerNorm(input_dim)
        elif method == "multihead_attention":
            # PyTorch's multi-head attention
            self.MultiHead_Attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
            self.layer_norm = nn.LayerNorm(input_dim)
            self.fc = nn.Linear(input_dim, input_dim)
    def forward(self, X):
        """
        Forward pass for aggregation.
        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, num_feats, num_points).
        Returns:
            torch.Tensor: Aggregated features of shape (batch_size, num_feats).
            torch.Tensor: Attention weights (if applicable).
        """
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
            X = X.transpose(1, 2)               # Shape: (batch_size, num_points, num_feats)
            self.scores = self.attn(X).squeeze(-1)   # Shape: (batch_size, num_points)
            # softmax for representing a valid probability distribution
            self.scores_softmax = torch.softmax(self.scores, dim=1).unsqueeze(1)  # Shape: (batch_size, 1, num_points)
            return torch.sum(X * self.scores_softmax, dim=2), self.scores
        elif self.method == "gated_attention":
            X = X.transpose(1, 2)   # Shape: (batch_size, num_points, num_feats)
            A_V = self.attn_V(X)    # Shape: (batch_size, num_points, hidden_dim)
            A_U = self.attn_U(X)    # Shape: (batch_size, num_points, hidden_dim)
            self.scores = self.attn_S(A_V * A_U).squeeze(-1)   # Shape: (batch_size, num_points)
            # softmax for representing a valid probability distribution
            self.scores_softmax = torch.softmax(self.A, dim=1).unsqueeze(1)  # Shape: (batch_size, 1, num_points)
            return torch.sum(X * self.scores_softmax, dim=2), self.scores
        elif self.method == "self_attention":
            # Custom self-attention
            X = X.transpose(1, 2)  # Shape: (batch_size, num_points, num_feats)
            # Compute Q, K, V
            Q = self.Query(X)  # Shape: (batch_size, num_points, hidden_dim)
            K = self.Key(X)    # Shape: (batch_size, num_points, hidden_dim)
            V = self.Value(X)  # Shape: (batch_size, num_points, hidden_dim)
            # Compute attention scores
            # dot products between queries and keys
            self.scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))
            # softmax for representing a valid probability distribution
            self.scores_softmax = F.softmax(self.scores, dim=-1)  # Shape: (batch_size, num_points, num_points)
            # Apply attention to values
            attn_output = torch.matmul(self.scores_softmax, V)    # Shape: (batch_size, num_points, hidden_dim)
            # Project back to input dimension
            attn_output = self.proj(attn_output)  # Shape: (batch_size, num_points, input_dim)
            # Residual connection and layer normalization
            attn_output = self.layer_norm(X + attn_output)
            # Aggregate across points (e.g., mean pooling)
            global_features = torch.mean(attn_output, dim=1)  # Shape: (batch_size, input_dim)
            return global_features, self.scores
        elif self.method == "multihead_attention":
            # PyTorch's multi-head attention
            X = X.transpose(1, 2)  # Shape: (batch_size, num_points, num_feats)
            X = X.transpose(0, 1)  # Shape: (num_points, batch_size, num_feats)
            # Apply multi-head attention
            attn_output, attn_weights = self.multihead_attn(X, X, X)  # Self-attention
            attn_output = attn_output.transpose(0, 1)  # Shape: (batch_size, num_points, num_feats)
            # Residual connection and layer normalization
            attn_output = self.layer_norm(X.transpose(0, 1) + attn_output)
            attn_output = self.fc(attn_output)  # Optional fully connected layer
            # Aggregate across points (e.g., mean pooling)
            global_features = torch.mean(attn_output, dim=1)  # Shape: (batch_size, num_feats)
            return global_features, attn_weights
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")


def train_model(model):
    model.train()


def eval_model(model):
    model.eval()


# ============================================================================
# # CLAM
# """
# Attention Network without Gating (2 fc layers)
# args:
#     L: input feature dimension
#     D: hidden layer dimension
#     dropout: whether to use dropout (p = 0.25)
#     n_classes: number of classes 
# """
# class Attn_Net(nn.Module):
#     def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
#         super(Attn_Net, self).__init__()
#         self.module = [
#             nn.Linear(L, D),
#             nn.Tanh()]
#         if dropout:
#             self.module.append(nn.Dropout(0.25))
#         self.module.append(nn.Linear(D, n_classes))
#         self.module = nn.Sequential(*self.module)
#     def forward(self, x):
#         return self.module(x), x # N x n_classes


# """
# Attention Network with Sigmoid Gating (3 fc layers)
# args:
#     L: input feature dimension
#     D: hidden layer dimension
#     dropout: whether to use dropout (p = 0.25)
#     n_classes: number of classes 
# """
# class Attn_Net_Gated(nn.Module):
#     def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
#         super(Attn_Net_Gated, self).__init__()
#         self.attention_a = [
#             nn.Linear(L, D),
#             nn.Tanh()]
#         self.attention_b = [nn.Linear(L, D),
#                             nn.Sigmoid()]
#         if dropout:
#             self.attention_a.append(nn.Dropout(0.25))
#             self.attention_b.append(nn.Dropout(0.25))
#         self.attention_a = nn.Sequential(*self.attention_a)
#         self.attention_b = nn.Sequential(*self.attention_b)
#         self.attention_c = nn.Linear(D, n_classes)
#     def forward(self, x):
#         a = self.attention_a(x)
#         b = self.attention_b(x)
#         A = a.mul(b)
#         A = self.attention_c(A)  # N x n_classes
#         return A, x

