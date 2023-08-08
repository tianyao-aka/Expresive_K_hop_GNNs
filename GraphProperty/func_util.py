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


def kl_diverrgence(u,std,anchor_u,anchor_std):
    val = torch.log(anchor_std)-torch.log(std)+(std**2+(u-anchor_u)**2)/(2*anchor_std**2)-0.5
    sample_kl = torch.mean(torch.sum(val,dim=1))
    return sample_kl


def bpr_loss_func(users_emb, pos_emb, neg_emb,userEmb0, posEmb0, negEmb0):
    N = users_emb.shape[0]*1.0
    reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                          posEmb0.norm(2).pow(2) +
                          negEmb0.norm(2).pow(2))
    reg_loss = reg_loss/N
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)
    loss = torch.mean(F.softplus(neg_scores - pos_scores))
    return loss, reg_loss


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


    def run(self,k_hop=[2,3],random_walk_order=10):
        walk_feature = self.calc_random_walk_feature(order=random_walk_order)
        for k in k_hop:
            t = self.calc_k_hop_neibrs(k)
            self.k_hop_neibrs.append(t.float())
        # normed_k_hop_adj = [self.postprocess_k_hop_neibrs(i.float()) for i in self.k_hop_neibrs]   # 是否使用D^-1*A
        return self.k_hop_neibrs,walk_feature



class Synthetic_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        print('heeloo')
        self.data, self.slices = torch.load(self.processed_paths[0])

    def create_new_data(self, data_list):
        labels = [i[0] for i in data_list]
        N = len(labels)
        X = torch.tensor([[1.0] * 5] * N)
        edges = []
        for idx, d in enumerate(data_list):
            for q in d[2:]:
                edges.append([idx, q])
                edges.append([q, idx])

        edge_index = torch.tensor(edges).t().contiguous()
        edge_index = torch.unique(edge_index, dim=1)
        k_hop_neibrs, random_walk_feats = EdgeIndex_Processor(edge_index).run()
        hop1, hop2, hop3 = k_hop_neibrs[0], k_hop_neibrs[1], k_hop_neibrs[2]
        hop1 = hop1.coalesce().indices().tolist()
        hop2 = hop2.coalesce().indices().tolist()
        hop3 = hop3.coalesce().indices().tolist()

        return Data(edge_index=edge_index, x=X, y=labels, rand_feature=random_walk_feats, hop1=hop1, hop2=hop2,
                    hop3=hop3)

    def collate_graph_adj(self, edge_list, ptr):
        edges = torch.cat([torch.tensor(i) + ptr[idx] for idx, i in enumerate(edge_list)], dim=1)
        N = ptr[-1]
        return torch.sparse_coo_tensor(edges, [1.] * edges.shape[1], (N, N))

    def process_data_list(self, file_path):
        with open(file_path, 'r') as f:
            data_list = []
            pyg_data_list = []
            for i in f.readlines():
                q = i.replace('\n', '').split(' ')
                q = [int(i) for i in q]
                if q[0] > 10:
                    if len(data_list) > 0:
                        pyg_data = self.create_new_data(data_list)
                        pyg_data_list.append(pyg_data)
                    data_list = []
                else:
                    data_list.append(q)
        return pyg_data_list

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        print('process')
        # Read data into huge `Data` list.
        data_list = self.process_data_list('data/TRIANGLE_EX/TRIANGLE_EX_test.txt')  # 记得修改这里的路径！
        data_list
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



    # def get_metric(self,idx):
    #     u = hu[idx].unsqueeze(0)
    #     out = torch.sum(u,hi,dim=-1)
    #     target_binary = [0]*91599
    #     for item_index in test_interactions[i]:
    # #             target[item_index]=rate
    #         target_binary[item_index]=1
    # #         target = torch.tensor(target)
    #     target_binary = torch.tensor(target_binary)
    #     seen = train_interations[i]
    #     out[seen]=-1.0
    #     ndcg_top20.append(retrieval_normalized_dcg(out,target_binary,k=20).item())
    # #         ndcg_top10.append(retrieval_normalized_dcg(pred,target,k=10).item())
    # #         ndcg_top30.append(retrieval_normalized_dcg(pred,target,k=30).item())
    # #         recall_at10.append(retrieval_recall(pred,target_binary,k=10).item())
    #     recall_at20.append(retrieval_recall(out,target_binary,k=20).item())
    # #         recall_at30.append(retrieval_recall(pred,target_binary,k=30).item())
    #     return recall_at20,ndcg_top20



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





