
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import random
import numpy as np
import pandas as pd
import scipy
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset


class ClassificationLoss(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_classes = num_class
    def forward(self, pred, label):
        return nn.CrossEntropyLoss()(pred, label)


transformation_function = nn.Softmax(dim=1)


class RegressionLoss(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_classes = num_class
    def forward(self, pred, label):
        return nn.MSELoss()(pred, label)


class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_class, scale=20.0, train_cutpoints=False):
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