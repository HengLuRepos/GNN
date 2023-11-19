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
        eye = torch.sparse_coo_tensor([range(all_num),range(all_num)],torch.ones(all_num), dtype=torch.float32, device=self.graph.device)
        user_weight = self.embedding_user.weight
        item_weight = self.embedding_item.weight
        embed = torch.cat([user_weight, item_weight])

        p0 = eye
        p1 = (self.a - self.b)/2 * eye + (self.a + self.b)/2 * self.graph

        embs = [torch.sparse.mm(eye, embed), torch.sparse.mm(p1, embed)]
        for k in range(2, self.layers + 1):
            theta_1 = (2*k + self.a + self.b) * (2*k + self.a + self.b - 1) / ((k + self.a + self.b) * 2*k)
            theta_2 = ((2*k + self.a + self.b - 1)*(self.a**2 - self.b**2)) /((2*k + self.a + self.b - 2) * (k + self.a + self.b) * 2 * k)
            emb_k = theta_1 * torch.sparse.mm(self.graph, embs[-1]) + theta_2 * embs[-1]
            theta_3 = ((k + self.a - 1) * (k + self.b - 1) * (2*k + self.a + self.b)) / (k*(self.a + self.b + k)*(2*k + self.a + self.b -2))
            emb_k -= theta_3 * embs[-2]
            embs.append(emb_k)
        band_stop = torch.stack(embs, dim=1).mean(dim=1)
        band_pass = torch.tanh(self.alpha * embed - band_stop)
        out = torch.hstack([band_stop, band_pass])
        users, items = torch.split(out, [self.num_user, self.num_item])
        return users, items
    def batch_bpr_loss(self, users, pos_items, neg_items):
        user_embs, item_embs = self.propagate()
        loss = 0.0
        for index, user in enumerate(users):
            user_emb = user_embs[user]
            pos_item = pos_items[index]
            pos_embs = item_embs[pos_item]
            neg_item = neg_items[index]
            neg_embs = item_embs[neg_item]
            yui = torch.mul(user_emb, pos_embs).sum(axis=1)
            yuj = torch.mul(user_emb, neg_embs).sum(axis=1, keepdim=True)
            loss += F.softplus(yuj - yui).mean()
        return loss
            


