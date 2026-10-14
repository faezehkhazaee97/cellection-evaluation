
"""
CELLECTION: A deep learning framework for phenotype prediction using Multiple Instance Learning
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Version information
__version__ = "0.1.3"

# Import main classes and functions
from .utils import (
    Create_MIL_Dataset,
    MILDataset,
    MIL_Collate_fn,
    MIL_Collate_fn_CPU_Inference
)

from .model import (
    PointNetBackbone,
    PointNetClassHead,
    CellEncoder,
    Tnet,
    SharedMLP,
    Aggregator,
    Conv1dBlock,
    FCBlock,
    FCNN,
    init_weights,
    set_seed
)

from .loss import (
    ClassificationLoss,
    RegressionLoss,
    OrdinalRegressionLoss,
    transformation_function
)

from .train import (
    cellectiion_object,
    train_model,
    eval_model
)

from .evaluator import *

# Main class for easy access
__all__ = [
    "cellectiion_object",
    "PointNetClassHead",
    "CellEncoder",
    "Create_MIL_Dataset",
    "MILDataset",
    "ClassificationLoss",
    "RegressionLoss",
    "OrdinalRegressionLoss",
    "__version__"
]
