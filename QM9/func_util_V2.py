import os
# import urllib.request
# from types import SimpleNamespace
# from urllib.error import HTTPError
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import tabulate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_geometric.utils import to_undirected,add_self_loops,remove_self_loops
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data,DataLoader
from torch_geometric.datasets import TUDataset




def collate_graph_adj(edge_list, ptr,use_gpu=False):
    if not use_gpu:
        edges = torch.cat([torch.tensor(i) + ptr[idx] for idx, i in enumerate(edge_list)], dim=1)
        N = ptr[-1]
        return torch.sparse_coo_tensor(edges,[1.]*edges.shape[1], (N, N))
    else:
        edges = torch.cat([torch.tensor(i).cuda(0) + ptr[idx] for idx, i in enumerate(edge_list)], dim=1)
        N = ptr[-1]
        val = torch.tensor([1.]*edges.shape[1]).cuda(0)
        return torch.sparse_coo_tensor(edges,val, (N, N)).cuda(0)


class EdgeIndex_Processor():
    def __init__(self, edge_index):
        super().__init__()
        self.random_walk = None
        adj,N = self.to_sparse_tensor(edge_index)
        adj_with_selfloop = self.to_sparse_tensor_with_selfloop(edge_index)
        self.N = N
        self.adj = adj.float()
        self.adj_with_loop = adj_with_selfloop.float()
        self.k_hop_neibrs = [adj.float()]
        self.calc_random_walk_matrix()

    def to_sparse_tensor(self, edge_index):
        edge_index = remove_self_loops(edge_index)[0]
        r = len(edge_index[0])
        N = edge_index.max() + 1
        t = torch.sparse_coo_tensor(edge_index, [1] * r, (N, N))
        return t, N

    def to_sparse_tensor_with_selfloop(self, edge_index):
        edge_index = add_self_loops(edge_index)[0]
        r = len(edge_index[0])
        N = edge_index.max() + 1
        t = torch.sparse_coo_tensor(edge_index, [1] * r, (N, N))
        return t

    def calc_random_walk_matrix(self):
        t = self.adj_with_loop.to_dense().sum(dim=1)
        t = 1./t
        n = len(t)
        ind = torch.tensor([[i,i] for i in range(n)]).T
        diag = torch.sparse_coo_tensor(ind,t,(n,n))
        random_walk = torch.sparse.mm(diag,self.adj)
        self.random_walk = random_walk

    def calc_random_walk_feature(self,order=10):
        t = self.random_walk
        tot_walk_feats = []
        walk_feats = []
        for i in range(self.N):
            walk_feats.append(t[i,i])
        tot_walk_feats.append(walk_feats)
        for i in range(order):
            walk_feats = []
            t = torch.sparse.mm(t,self.random_walk)
            for i in range(self.N):
                walk_feats.append(t[i, i])
            tot_walk_feats.append(walk_feats)
        tot_walk_feats = torch.tensor(tot_walk_feats).T
        return tot_walk_feats


    def calc_adj_power(self,adj, power):
        t = adj
        for _ in range(power - 1):
            t = torch.sparse.mm(t, adj)
        # set value to one
        indices = t.coalesce().indices()
        v = t.coalesce().values()
        v = torch.tensor([1 if i > 1 else i for i in v])
        diag_mask = indices[0] != indices[1]
        indices = indices[:, diag_mask]
        v = v[diag_mask]
        t = torch.sparse_coo_tensor(indices, v, (self.N, self.N))
        return t

    def postprocess_k_hop_neibrs(self,sparse_adj):
        diag = torch.diag(1. / sparse_adj.to_dense().sum(dim=1))
        diag = diag.to_sparse()
        out = torch.sparse.mm(diag, sparse_adj)
        return out


    def calc_k_hop_neibrs(self,k_hop=2):
        adj_hop_k = self.calc_adj_power(self.adj, k_hop)
        one_hop = self.k_hop_neibrs[0]
        prev_hop = self.k_hop_neibrs[1:k_hop]
        for p in prev_hop:
            one_hop += p
        final_res = adj_hop_k - one_hop

        indices = final_res.coalesce().indices()
        v = final_res.coalesce().values()
        v = [0 if i <= 0 else 1 for i in v]
        masking = []
        v_len = len(v)
        for i in range(v_len):
            if v[i] > 0:
                masking.append(i)
        v = torch.tensor(v)
        masking = torch.tensor(masking).long()
        indices = indices[:, masking]
        v = v[masking]
        final_res = torch.sparse_coo_tensor(indices, v, (self.N, self.N))
        return final_res


    def run(self,k_hop=[2,3,4,5,6],random_walk_order=20):
        walk_feature = self.calc_random_walk_feature(order=random_walk_order)
        for k in k_hop:
            t = self.calc_k_hop_neibrs(k)
            self.k_hop_neibrs.append(t.float())
        # normed_k_hop_adj = [self.postprocess_k_hop_neibrs(i.float()) for i in self.k_hop_neibrs]   # 是否使用D^-1*A
        return self.k_hop_neibrs,walk_feature



def transform(t):
    q, j = EdgeIndex_Processor(t.edge_index).run()
    hop1, hop2, hop3, hop4, hop5, hop6 = q[0], q[1], q[2], q[3], q[4], q[5]
    t.rand_feature = j
    x2 = torch.concat((t.x, j), dim=1)
    hop1_feature = hop1.matmul(x2)
    hop2_feature = hop2.matmul(x2)
    hop3_feature = hop3.matmul(x2)
    hop4_feature = hop4.matmul(x2)
    hop5_feature = hop5.matmul(x2)
    hop6_feature = hop6.matmul(x2)

    hop1 = hop1.coalesce().indices().tolist()
    hop2 = hop2.coalesce().indices().tolist()
    hop3 = hop3.coalesce().indices().tolist()
    hop4 = hop4.coalesce().indices().tolist()
    hop5 = hop5.coalesce().indices().tolist()
    hop6 = hop6.coalesce().indices().tolist()
    t.hop1 = hop1
    t.hop2 = hop2
    t.hop3 = hop3
    t.hop4 = hop4
    t.hop5 = hop5
    t.hop6 = hop6
    t.hop1_feature = hop1_feature
    t.hop2_feature = hop2_feature
    t.hop3_feature = hop3_feature
    t.hop4_feature = hop4_feature
    t.hop5_feature = hop5_feature
    t.hop6_feature = hop6_feature
    return t


if __name__=='__main__':
    pass
    # edges = torch.tensor([[0, 1, 0, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 1, 3, 2]]).long()
    # data_model = EdgeIndex_Processor(edges)
    # q,j = data_model.run()
    # print (q[0])
    # print (j)
    # s = Synthetic_Dataset(root='data/pyg_TRIANGLE_EX/test')
    # for d in s:
    #     if max(d.y)>1:
    #         print (d.y)





