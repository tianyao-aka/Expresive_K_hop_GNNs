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
from torch_geometric import transforms as T
from torch_geometric.utils import degree
from sklearn.model_selection import StratifiedKFold, KFold
from time import time


def weight_reset(m):
    if isinstance(m, nn.Module) and hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def khop_transform(data):
    s = time()
    try:
        N = data.x.shape[0]
        num_features = data.x.shape[1]
    except:
        N = data.edge_index.max() + 1
        num_features = 0
    preprocessor = EdgeIndex_Processor(data.edge_index, N)
    k_hop_mat, rw_feature,central_to_subgraph_stats,context_same_hop_stats = preprocessor.run()
    hop1, hop2, hop3, hop4, hop5 = k_hop_mat[0], k_hop_mat[1], k_hop_mat[2], k_hop_mat[3], k_hop_mat[4]
    hop1 = hop1.coalesce().indices().tolist()
    hop2 = hop2.coalesce().indices().tolist()
    hop3 = hop3.coalesce().indices().tolist()
    hop4 = hop4.coalesce().indices().tolist()
    hop5 = hop5.coalesce().indices().tolist()
    data.hop1 = hop1
    data.hop2 = hop2
    data.hop3 = hop3
    data.hop4 = hop4
    data.hop5 = hop5

    data.rw_feature = rw_feature
    data.nfeats = num_features
    data.central_to_subgraph_feats = central_to_subgraph_stats # root node of subgraph to K-hop landing probability stats
    data.context_samehop_feats = context_same_hop_stats # k-hop node in a subgraph to same k-hop nodes landing probability stats
    # print ('output context and central feats')
    t = time()
    return data

def degree_post_processing(data):
    deg_func = T.OneHotDegree(max_degree=10000)
    data = deg_func(data)
    num_feats = data.nfeats
    N = data.x.shape[0]
    degrees = degree(data.edge_index[0],num_nodes=N).view(-1,1).float()
    max_degree = degrees.max().item()
    degrees = degrees/max_degree
    # print ('num feats:',num_feats)
    if num_feats>0:
        feature = data.x[:,:num_feats]
        deg_feats = data.x[:,num_feats:]
        val = torch.cat([degrees,deg_feats],dim=1)
        val = val[:,:65]
        data.x = torch.cat((feature,val),dim=1)
    else:
        val = torch.cat([degrees,data.x],dim=1)
        val = val[:,:65]
        data.x = val
    return data

def pre_cache_khop_features(data):
    s = time()
    concat_feature = torch.cat((data.x,data.rw_feature),dim=1)
    for i in range(1,5+1):
        mat = torch.tensor(data[f'hop{i}']).float()
        r = len(mat[0])
        N = data.x.shape[0]
        t = torch.sparse_coo_tensor(mat, [1.] * r, (N, N))
        feats = torch.sparse.mm(t,concat_feature)
        data[f'hop{i}_features'] = feats
    t = time()
    return data

def Composer(*funcs):
    val = [f for f in funcs]
    return T.Compose(val)



def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def k_fold_without_validation(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for tr_idx, test_idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        train_indices.append(torch.from_numpy(tr_idx))
        test_indices.append(torch.from_numpy(test_idx))
    return train_indices, test_indices

def kl_diverrgence(u,std,anchor_u,anchor_std):
    val = torch.log(anchor_std)-torch.log(std)+(std**2+(u-anchor_u)**2)/(2*anchor_std**2)-0.5
    sample_kl = torch.mean(torch.sum(val,dim=1))
    return sample_kl


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


def collate_edge_index(edge_list, ptr,use_gpu=False):
    if not use_gpu:
        edges = torch.cat([torch.tensor(i) + ptr[idx] for idx, i in enumerate(edge_list)], dim=1)
        return edges
    else:
        edges = torch.cat([torch.tensor(i).cuda(0) + ptr[idx] for idx, i in enumerate(edge_list)], dim=1)
        return edges



class EdgeIndex_Processor():
    def __init__(self, edge_index, num_nodes):
        super().__init__()
        # print (f'number of nodes:{num_nodes}, number of edges:{edge_index.shape[1]}')
        self.power_adj = []
        self.random_walk = None
        self.N = num_nodes
        adj, N = self.to_sparse_tensor(edge_index)
        self.adj_with_selfloop = self.to_sparse_tensor_with_selfloop(edge_index).float()
        self.adj = adj.float()
        # self.adj_with_loop = adj_with_selfloop.float()
        self.power_adj.append(self.adj)
        self.k_hop_neibrs = [adj.float()]
        self.calc_random_walk_matrix()  # use adj with self loop
        self.calc_power_adj(hop=5)  # calculate to 5th hop


    def to_sparse_tensor(self, edge_index):
        edge_index = remove_self_loops(edge_index)[0]
        r = len(edge_index[0])
        N = self.N
        t = torch.sparse_coo_tensor(edge_index, [1] * r, (N, N))
        return t, N

    def to_sparse_tensor_with_selfloop(self, edge_index):
        edge_index = add_self_loops(edge_index)[0]
        r = len(edge_index[0])
        N = self.N
        t = torch.sparse_coo_tensor(edge_index, [1] * r, (N, N))
        return t

    def calc_power_adj(self,hop):
        # calculate adj without self loop
        adjs = []
        t =self.adj
        for _ in range(hop-1):
            t = torch.sparse.mm(t,self.adj)
            self.power_adj.append(t)


    def calc_random_walk_matrix(self):
        diag_val = []
        for i in range(self.N):
            res = torch.sum(self.adj_with_selfloop[i].coalesce().values())
            diag_val.append(res)
        t = torch.tensor(diag_val)
        # t = self.adj.to_dense().sum(dim=1)
        t = 1. / t
        t[torch.isinf(t)] = 0.
        n = self.N
        ind = torch.tensor([[i, i] for i in range(n)]).T
        diag = torch.sparse_coo_tensor(ind, t, (n, n))
        random_walk = torch.sparse.mm(self.adj_with_selfloop,diag)
        self.random_walk = random_walk


    def calc_random_walk_feature(self, order=32):
        rw_stats = []
        n = self.N
        ind = torch.tensor([[i, i] for i in range(self.N)]).T
        rw_val = torch.sparse_coo_tensor(ind, torch.tensor([1.]*self.N), (n, n))
        t = self.random_walk
        for i in range(order):
            rw_val = torch.sparse.mm(t,rw_val)
            rw_stats.append(rw_val)
        return rw_stats

    def self_return_prob(self,rw_info):
        total_self_prob = []
        for mat in rw_info:
            k_step_prob = []
            for i in range(self.N):
                prob = mat[i,i]
                k_step_prob.append(prob)
            total_self_prob.append(k_step_prob)
        res = torch.tensor(total_self_prob).T
        return res


    def calc_k_hop_neibrs(self):
        self_ind = set([(i,i) for i in range(self.N)])
        N = len(self.power_adj)  # total hop count
        for p in range(1,N):
            prev_accumu_hop = torch.sparse_coo_tensor([[1, 0], [0, 1]], [0., 0.], size=(self.N, self.N))
            prev_hop = self.k_hop_neibrs[:p]   # don't use +=, this is in place opt, which destroy the program!
            cur_hop = self.power_adj[p]
            for mat in prev_hop:
                prev_accumu_hop = prev_accumu_hop + mat
            ind = prev_accumu_hop.coalesce().indices()
            v = prev_accumu_hop.coalesce().values()
            indices = ind[:,v>0]
            val = indices.T.numpy().tolist()
            prev_indices = set([tuple(i) for i in val])
            prev_indices.update(self_ind)
            cur_index = cur_hop.coalesce().indices().T.numpy().tolist()
            cur_index = set([tuple(i) for i in cur_index])
            cur_index = cur_index-prev_indices
            num_nodes = len(cur_index)
            if num_nodes>0:
                val = [1.]*num_nodes
                cur_index = torch.tensor([list(i) for i in cur_index]).T
                hopK_mat = torch.sparse_coo_tensor(cur_index,val,size=(self.N,self.N))
                self.k_hop_neibrs.append(hopK_mat)
            else:
                hopK_mat = torch.sparse_coo_tensor([[0,1],[1,0]], [0,0], size=(self.N, self.N))
                self.k_hop_neibrs.append(hopK_mat)


    # input A_k, RW from 1 to L
    def calc_context_structure_prob(self,hopk_matrix, rw_info, N):
        # hopk_matrix:  a sparse_coo_tensor holding hop_K matrix
        # ranom_walk_matrix_list: a list of random walk sparse matrix
        # N: int, number of nodes in the graph
        hopK_rw_feature_center_to_hopk = []
        hopK_rw_feature_samehop_hopk = []
        for n in range(N):
            val1 = []
            val2 = []
            hopk_nodes = hopk_matrix[n].coalesce().indices().view(-1, ).numpy().tolist()
            if len(hopk_nodes)>40:
                hopk_nodes = hopk_nodes[:40]
            for rw in rw_info:
                if len(hopk_nodes) == 0:
                    val1.append(0.)
                    val2.append(0.)
                    continue
                rw = rw.to_dense()
                # calc  root to central stats
                hopK_probs = torch.sum(rw[hopk_nodes,n])
                val1.append(hopK_probs)
                # calc same-hop stats
                idx = torch.tensor([(i,j) for i in hopk_nodes for j in hopk_nodes]).long()
                hopK_probs = torch.sum(rw[idx[:,0],idx[:,1]])/len(hopk_nodes)
                val2.append(hopK_probs)
            hopK_rw_feature_center_to_hopk.append(val1)
            hopK_rw_feature_samehop_hopk.append(val2)
        return torch.tensor(hopK_rw_feature_center_to_hopk),torch.tensor(hopK_rw_feature_samehop_hopk)

    def calc_agg_context_structure_nodes(self,hopk_matrix_list, rw_info, N):
        output_central = []
        output_samehop = []
        for hopk in hopk_matrix_list:
            ret_central,ret_samehop = self.calc_context_structure_prob(hopk,rw_info,N)
            ret_central = ret_central.unsqueeze(0)
            ret_samehop = ret_samehop.unsqueeze(0)
            output_central.append(ret_central)
            output_samehop.append(ret_samehop)
        output_central = torch.cat(output_central, dim=0)
        output_samehop = torch.cat(output_samehop, dim=0)
        return torch.mean(output_central, dim=0),torch.mean(output_samehop, dim=0)


    def run(self, random_walk_order=32):
        s = time()
        self.calc_k_hop_neibrs()
        t1 = time()
        # normed_k_hop_adj = [self.postprocess_k_hop_neibrs(i.float()) for i in self.k_hop_neibrs]   # 是否使用D^-1*A
        rw_stats = self.calc_random_walk_feature()
        t2 = time()
        self_return_features = self.self_return_prob(rw_stats)   # calc self-return probability
        t3 = time()
        central_to_subgraph_features,same_hop_in_subgraph_stats_features = self.calc_agg_context_structure_nodes(self.k_hop_neibrs,rw_stats,self.N)
        t4 = time()
        # same_hop_in_subgraph_stats_features = self.calc_agg_rw_stats_context_to_same_hop_nodes(self.k_hop_neibrs,rw_stats,self.N)
        # t5 = time()
        return self.k_hop_neibrs,self_return_features,central_to_subgraph_features,same_hop_in_subgraph_stats_features



if __name__=='__main__':
    edges = torch.tensor([[0, 1, 1,2,2,3,2,4,3,4], [1, 0,2,1,3,2,4,2,4,3]]).long()
    data_model = EdgeIndex_Processor(edges,5)
    q = data_model.run()
    print (1)
    # print (q[0])
    # print (j)
    # s = Synthetic_Dataset(root='data/pyg_TRIANGLE_EX/test')
    # for d in s:
    #     if max(d.y)>1:
    #         print (d.y)
