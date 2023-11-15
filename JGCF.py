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

        embs = [torch.sparse.mm(eye, embed), torch.sparse.mm(p1, embed)]
        for k in range(2, self.layers + 1):
            theta_1 = (2*k + self.a + self.b) * (2*k + self.a + self.b - 1) / ((k + self.a + self.b) * 2*k)
            theta_2 = ((2*k + self.a + self.b - 1)*(self.a**2 - self.b**2)) /((2*k + self.a + self.b - 2) * (k + self.a + self.b) * 2 * k)
            emb_k = theta_1 * torch.sparse.mm(self.graph, embs[-1]) + theta_2 * embs[-1]
            theta_3 = ((k + self.a - 1) * (k + self.b - 1) * (2*k + self.a + self.b)) / (k*(self.a + self.b + k)*(2*k + self.a + self.b -2))
            emb_k = -theta_3 * embs[-2]
            embs.append(emb_k)
        band_stop = torch.stack(embs, dim=1).mean(dim=1)
        band_pass = torch.tanh(self.alpha * embed - band_stop)
        out = torch.hstack([band_stop, band_pass])
        users, items = torch.split(out, [self.num_user, self.num_item])
        return users, items

            


