import torch
import torch.nn as nn
import os
import numpy as np
import scipy.sparse as sp

class Dataset:
    def __init__(self, path, train=True):
        file = path + 'train.txt' if train else path + 'test.txt'
        self.num_user = 0
        self.num_item = 0
        self.users = []
        self.unique_users = []
        self.items = []
        with open(file) as f:
            for line in f.readlines():
                if len(line) > 0:
                    line = line.strip('\n').split(' ')
                    userId = int(line[0])
                    item = [int(id) for id in line[1:]]
                    self.num_user = max(self.num_user, userId)
                    self.num_item = max(self.num_item, max(item))
                    self.unique_users.append(userId)
                    self.items.extend(item)
                    self.users.extend([userId] * len(item))
        self.users = np.array(self.users)
        self.unique_users = np.array(self.unique_users)
        self.items = np.array(self.items)
        self.num_user += 1
        self.num_item += 1

        self.userItemMatrix = sp.csr_matrix((np.ones(len(self.users)), 
                                            (self.users, self.items)), 
                                            shape=(self.num_user,self.num_item))
        
        self.Graph = None
    def _sp_to_torch(self,x):
        coo = x.tocoo().astype(np.float32)
        row = torch.LongTensor(coo.row)
        col = torch.LongTensor(coo.col)
        ind = torch.stack([row,col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index,data, torch.Size(coo.shape))
    def build_graph(self):
        adj_mat = sp.dok_matrix((self.num_item + self.num_user, self.num_item + self.num_user), dtype=np.float_32)
        adj_mat = adj_mat.tolil()
        R = self.userItemMatrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(adj_mat)
        norm_adj = norm_adj.tocsr()

        self.Graph = self._sp_to_torch(norm_adj)
        self.Graph = self.Graph.coalesce()
    def get_graph(self):
        return self.Graph
    def get_user_number(self):
        return self.num_user
    def get_item_number(self):
        return self.num_item

class LightGCN(nn.Module):
    def __init__(self, dataset: Dataset, config):
        super().__init__()
        self.num_user = dataset.get_user_number()
        self.num_item = dataset.get_item_number()
        self.graph = dataset.get_graph()
        self.embedding_dim = config.embedding_dim
        self.lr = config.lr
        self.layers = config.layers
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim)
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim)

    def propagate(self):
        user_weight = self.embedding_user.weight
        item_weight = self.embedding_item.weight
        embed = torch.cat([user_weight, item_weight])
        embs = [embed]
        for layer in self.layers:
            embed = torch.sparse.mm(self.graph, embed)
            embs.append(embed)
        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        users, items = torch.split(out, [self.num_user, self.num_item])
        return users, items
    
    