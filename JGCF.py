import torch
import torch.nn as nn
import os
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from lightGCN import Dataset, LightGCN

class JGCF(LightGCN):
    def __init__(self, dataset: Dataset, config):
        super().__init__(dataset, config)
        self.a = config.a
        self.b = config.b
        self.alpha = config.alpha
    def propagate(self):
        all_num = self.num_user + self.num_item
        eye = torch.sparse_coo_tensor([range(all_num),range(all_num)],torch.ones(all_num), dtype=torch.float32)
        user_weight = self.embedding_user.weight
        item_weight = self.embedding_item.weight
        embed = torch.cat([user_weight, item_weight])

        p0 = eye.to_sparse_csr()
        p1 = (self.a - self.b)/2 * eye + (self.a + self.b)/2 * self.graph

        pk = [p0, p1.to_sparse_csr()]
        for k in range(2, self.layers + 1):
            theta_1 = (2*k + self.a + self.b) * (2*k + self.a + self.b - 1) / ((k + self.a + self.b) * 2*k)
            theta_2 = ((2*k + self.a + self.b - 1)*(self.a**2 - self.b**2)) /((2*k + self.a + self.b - 2) * (k + self.a + self.b) * 2 * k)
            temp = theta_1*self.graph + theta_2*eye
            p = torch.sparse.mm(temp.to_sparse_csr(), pk[-1]).to_sparse_csr()
            del temp
            theta_3 = ((k + self.a - 1) * (k + self.b - 1) * (2*k + self.a + self.b)) / (k*(self.a + self.b + k)*(2*k + self.a + self.b -2))
            temp = -theta_3 * pk[-2]
            p = p + temp.to_sparse_csr()
            pk.append(p.to_sparse_csr())
        band_stop = torch.stack([torch.sparse.mm(p, embed) for p in pk], dim=1).mean(dim=1)
        band_pass = torch.tanh(self.alpha * embed - band_stop)
        out = torch.hstack([band_stop, band_pass])
        users, items = torch.split(out, [self.num_user, self.num_item])
        return users, items

            


