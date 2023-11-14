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
        self.noise_norm = config.noise_norm
        self.lam = config.lam
        self.tau = config.tau
    def noise_propagate(self):
        user_weight = self.embedding_user.weight
        item_weight = self.embedding_item.weight
        embed = torch.cat([user_weight, item_weight])

        embs = []
        for layer in range(self.layers):
            prod = torch.sparse.mm(self.graph, embed)
            noise = torch.rand(prod.shape)
            norms = torch.norm(noise, p=2, dim=1, keepdim=True)
            noise /= norms
            noise *= torch.sign(prod) * self.noise_norm
            embs.append(prod + noise)
        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        users, items = torch.split(out, [self.num_user, self.num_item])
        return users, items
    def cl_loss(self):
        user_noise_1, items_noise_1 = self.noise_propagate()
        user_noise_2, items_noise_2 = self.noise_propagate()
        
        