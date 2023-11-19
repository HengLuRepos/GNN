import torch
import torch.nn as nn
import os
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from lightGCN import Dataset, LightGCN

class SimGCL(LightGCN):
    def __init__(self, dataset: Dataset, config):
        super().__init__(dataset, config)
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
            embed = prod + noise
            embs.append(embed)
        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        users, items = torch.split(out, [self.num_user, self.num_item])
        return users, items
    def cl_loss(self):
        user_noise_1, items_noise_1 = self.noise_propagate()
        user_noise_2, items_noise_2 = self.noise_propagate()

        user_noise_1 /= torch.norm(user_noise_1, p=2, dim=1, keepdim=True)
        user_noise_2 /= torch.norm(user_noise_2, p=2, dim=1, keepdim=True)

        pos_user_pairs = torch.sum(user_noise_1 * user_noise_2, axis=1)
        pos_user_pairs = torch.exp(pos_user_pairs / self.tau) 
        all_user_pairs = torch.matmul(user_noise_1, user_noise_2.T)
        all_user_pairs = torch.exp(all_user_pairs / self.tau).sum(axis=1)
        cl_user_loss = torch.sum(-torch.log(pos_user_pairs/(all_user_pairs - pos_user_pairs)))


        items_noise_1 /= torch.norm(items_noise_1, p=2, dim=1, keepdim=True)
        items_noise_2 /= torch.norm(items_noise_2, p=2, dim=1, keepdim=True)

        pos_item_pairs = torch.sum(items_noise_1 * items_noise_2, axis=1)
        pos_item_pairs = torch.exp(pos_item_pairs / self.tau)
        all_item_pairs = torch.matmul(items_noise_1, items_noise_2.T)
        all_item_pairs = torch.exp(all_item_pairs / self.tau).sum(axis=1)
        cl_item_loss = torch.sum(-torch.log(pos_item_pairs/(all_item_pairs - pos_item_pairs)))

        return cl_user_loss + cl_item_loss
    
    def batch_cl_loss(self, users, items):
        user_noise_1, items_noise_1 = self.noise_propagate()
        user_noise_2, items_noise_2 = self.noise_propagate()

        user_noise_1 = user_noise_1[users]
        user_noise_2 = user_noise_2[users]
        user_noise_1_norm = user_noise_1 / torch.norm(user_noise_1, p=2, dim=1, keepdim=True)
        user_noise_2_norm = user_noise_2 / torch.norm(user_noise_2, p=2, dim=1, keepdim=True)

        pos_user_pairs = torch.sum(user_noise_1_norm * user_noise_2_norm, axis=1)
        pos_user_pairs = torch.exp(pos_user_pairs / self.tau) 
        all_user_pairs = torch.matmul(user_noise_1_norm, user_noise_2_norm.T)
        all_user_pairs = torch.exp(all_user_pairs / self.tau).sum(axis=1)
        cl_user_loss = torch.sum(-torch.log(pos_user_pairs/(all_user_pairs - pos_user_pairs)))

        items_noise_1 = items_noise_1[items]
        items_noise_2 = items_noise_2[items]
        items_noise_1_norm = items_noise_1 / torch.norm(items_noise_1, p=2, dim=1, keepdim=True)
        items_noise_2_norm = items_noise_2 / torch.norm(items_noise_2, p=2, dim=1, keepdim=True)

        pos_item_pairs = torch.sum(items_noise_1_norm * items_noise_2_norm, axis=1)
        pos_item_pairs = torch.exp(pos_item_pairs / self.tau)
        all_item_pairs = torch.matmul(items_noise_1_norm, items_noise_2_norm.T)
        all_item_pairs = torch.exp(all_item_pairs / self.tau).sum(axis=1)
        cl_item_loss = torch.sum(-torch.log(pos_item_pairs/(all_item_pairs - pos_item_pairs)))

        return self.lam* (cl_user_loss + cl_item_loss)

