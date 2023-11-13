import torch
import torch.nn as nn
import os
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from lightGCN import Dataset, LightGCN

class SimGCL(LightGCN):
    def __init__(self, dataset: Dataset, config):
        super.__init__(dataset, config)
        