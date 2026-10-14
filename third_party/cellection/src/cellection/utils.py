from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .loss import *

import os
import copy
import random

import scipy
import numpy as np
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


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
        ins.append(len(indices))                    # Number of instances in each bag
        metadata_indices.append(indices)            # Save metadata indices for each bag
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
        bag = torch.FloatTensor(self.Xs[idx]).to(self.device)                   # Convert bag to tensor and move to device
        label = torch.tensor(self.Ys[idx], dtype=torch.long).to(self.device)    # Convert label to tensor and move to device
        ins_ct = self.ins[idx]                                                  # Number of instances in the bag
        metadata_idx = self.metadata_indices[idx]                               # Metadata indices for the bag
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
    max_length = max(ins)               # Set maximum number of instances in the batch for padding bags to the same length
    padded_bags = torch.zeros((len(bags), max_length, bags[0].shape[1]), dtype=torch.float32)
    for i, bag in enumerate(bags):
        padded_bags[i, :len(bag)] = bag
    labels = torch.stack(labels)        # Stack labels into a single tensor
    return padded_bags.to(device), labels.to(device), lengths, metadata_idx


def MIL_Collate_fn_CPU_Inference(batch):
    """
    Similar to MIL_Collate_fn, but during inference, we have option to switch to CPU
    """
    bags, labels, ins, metadata_idx= zip(*batch)
    lengths = ins
    max_length = max(ins)                            
    padded_bags = torch.zeros((len(bags), max_length, bags[0].shape[1]), dtype=torch.float32)
    for i, bag in enumerate(bags):
        padded_bags[i, :len(bag)] = bag
    labels = torch.stack(labels)        
    return padded_bags.to('cpu'), labels.to('cpu'), lengths, metadata_idx