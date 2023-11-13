import torch
import torch.nn as nn
import os
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
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
                                            shape=(self.num_user,self.num_item), dtype=np.float32)
        
        self.Graph = None
        self.build_graph()
    def _sp_to_torch(self,x):
        coo = x.tocoo().astype(np.float32)
        row = torch.LongTensor(coo.row)
        col = torch.LongTensor(coo.col)
        ind = torch.stack([row,col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(ind,data, torch.Size(coo.shape)).to_sparse_csr()
    def build_graph(self):

        R = self.userItemMatrix.tocsr().astype(np.float32)
        upper = sp.csc_matrix((self.num_user, self.num_user), dtype=np.float32)
        upper = sp.hstack([upper, R.tocsc()])
        lower = sp.csc_matrix((self.num_item, self.num_item),dtype=np.float32)
        lower = sp.hstack([R.T.tocsc(), lower])
        adj_mat = sp.vstack([upper.tocsr(), lower.tocsr()])
        #adj_mat = sp.bmat([[None, R], [R.T.tocsr().astype(np.float32), None]], format="csr", dtype=np.float32)


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
    def get_positive_items(self, users):
        positive_items = []

        for user in users:
            pos_item = self.userItemMatrix[user].nonzero()[1]
            positive_items.append(pos_item)

        return positive_items

class LightGCN(nn.Module):
    def __init__(self, dataset: Dataset, config):
        super().__init__()
        self.dataset = dataset
        self.num_user = dataset.get_user_number()
        self.num_item = dataset.get_item_number()
        self.graph = nn.Parameter(dataset.get_graph(), requires_grad=False)
        self.embedding_dim = config.embedding_dim
        self.lr = config.lr
        self.layers = config.layers
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim)
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim)
        nn.init.normal_(self.embedding_user.weight)
        nn.init.normal_(self.embedding_item.weight)

    def propagate(self):
        user_weight = self.embedding_user.weight
        item_weight = self.embedding_item.weight
        embed = torch.cat([user_weight, item_weight])
        
        embs = [embed]
        for layer in range(self.layers):
            embed = torch.sparse.mm(self.graph, embed)
            embs.append(embed)
        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        users, items = torch.split(out, [self.num_user, self.num_item])
        return users, items
    def forward(self, users, items):
        user_emb, item_emb = self.propagate()

        user_emb = user_emb[users]
        item_emb = item_emb[items]

        scores = torch.mul(user_emb, item_emb).sum(dim=1)
        return scores
    def bpr_loss(self, users, pos_items, neg_items):
        user_embs, item_embs = self.propagate()
        print(self.graph[0])
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

            

    