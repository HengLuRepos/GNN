import torch
import torch.nn as nn
import os
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from lightGCN import Dataset, LightGCN
from simGCL import SimGCL

class CJGCF(SimGCL):
    def __init__(self, dataset: Dataset, config):
        super().__init__(dataset, config)
        self.a = config.a
        self.b = config.b
        self.alpha = config.alpha
    