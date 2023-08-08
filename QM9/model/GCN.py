from torch_geometric.nn import SAGEConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
import sys
sys.path.append('..')
# from nov.dataset_processing import Processed_Dataset
from dataset_processing import Processed_Dataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import torch.nn as nn
from torch_scatter import scatter_mean,scatter_sum
from torch_geometric.nn import MessagePassing,global_add_pool,global_mean_pool
from combine import GeometricCombine

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree,remove_self_loops
from copy import copy
from func_util import collate_edge_index,weight_reset

use_cuda =True if torch.cuda.is_available() else False
gpu_num = 1 if torch.cuda.is_available() else None


class QM9InputEncoder(nn.Module):
    def __init__(self, hidden_size,rw_size, use_pos=False):
        super(QM9InputEncoder, self).__init__()
        self.use_pos = use_pos
        if use_pos:
            input_size = 22
        else:
            input_size = 19
        self.init_proj = nn.Linear(input_size+rw_size+1, hidden_size)
        self.z_embedding = nn.Embedding(1000, 8)

    def reset_parameters(self):
        self.init_proj.reset_parameters()
        self.z_embedding.reset_parameters()

    def forward(self, data):
        x = data.x
        z = data.z
        if "pos" in data:
            pos = data.pos
        else:
            pos = None

        z_emb = 0
        if z is not None:
            ### computing input node embedding
            z_emb = self.z_embedding(data.z)
            if z_emb.ndim == 3:
                z_emb = z_emb.sum(dim=1)
        # concatenate with continuous node features
        x = torch.cat([z_emb, x], -1)

        if self.use_pos:
            x = torch.cat([x, pos], 1)

        x = self.init_proj(x)
        return x



class GCNE_Model(nn.Module):
    def __init__(self,data,dim=128,dropout=0.5,k=4,separate_hop_conv=True,combine='geometric',layers = 2,feature_fusion='average',JK='concat',normed=True):
        super().__init__()
        self.k = k
        in_dim = data.num_features+data[0].rand_feature.shape[1]
        self.node_emb = QM9InputEncoder(hidden_size=dim,rw_size=21)
        self.edge_emb = nn.Linear(data.num_edge_features,dim,bias = False)
        self.JK = JK
        self.convs = nn.ModuleList([GCNE_Block(dim,dropout, k, separate_hop_conv,combine,normed=normed) for _ in range(layers)])
        self.feature_fusion = feature_fusion
        self.alphas = nn.Parameter(torch.zeros(1,3,data[0].rand_feature.shape[1]))
        self.layers = layers
        self.normed_degree = normed
        if self.JK=='concat':
            self.JK_layer = nn.Linear((layers+1)*dim,dim)
        self.apply(weight_reset)

    def forward(self, data):
        batch = data.batch
        if 'central_to_subgraph_feats' in data:
            x,rw,central_to_subgraph,samehop = data.x,data.rw_feature,data.central_to_subgraph_feats,data.context_samehop_feats
            if self.feature_fusion == 'average':
                avg_feats = (rw+central_to_subgraph+samehop)/3.0
            else:
                f = torch.cat([rw.unsqueeze(1),central_to_subgraph.unsqueeze(1),samehop.unsqueeze(1)],dim=1)   # N*3*F
                w = F.softmax(self.alphas,dim=1)
                avg_feats = torch.sum(f*w,dim=1)   # N*F
            x = torch.cat((x,avg_feats),dim=1)
            data.x = x
        else:
            # only rw features
            x, rw = data.x, data.rand_feature
            x = torch.cat((x,rw),dim=1)
            data.x = x
        x = self.node_emb(data)
        data.x = x
        out = [x]
        if 'edge_attr' in data:
            e = data.edge_attr
            e = self.edge_emb(e)
            data.edge_attr = e
        else:
            data.edge_attr = None
        for l in range(self.layers):
            h = self.convs[l](data)
            out.append(h.x)
        if self.JK =='last':
            return global_add_pool(h.x,batch)
        else:
            out = torch.cat([i for i in out],dim=-1)
            return global_add_pool(self.JK_layer(out),batch)


class GCNE_Block(nn.Module):
    def __init__(self,dim=128,dropout=0.5,k=4,separate_hop_conv=True,combine='geometric',normed=True):
        super().__init__()
        self.k = k
        self.hop_ind = [f'hop{i+1}' for i in range(k)]
        self.separate_conv = separate_hop_conv
        if separate_hop_conv:
            self.convs = nn.ModuleList([GCNEConv(dim,dim,normed=normed) for _ in range(k)])
        else:
            self.convs = nn.ModuleList([GCNEConv(dim,dim,normed=normed)])
        if combine=='geometric':
            self.combine = GeometricCombine(k,dim)
        else:
            self.combine = 'add'
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        data = new(data)
        hop_matrices = [collate_edge_index(data[idx],data.ptr,use_cuda) for idx in self.hop_ind]
        out = []
        if 'edge_attr' in data:
            edge_attr = data.edge_attr
            val,edge_attr = self.convs[0](data.x,data.edge_index,edge_attr)
            out.append(val)
            data.edge_attr = edge_attr
        else:
            val = self.convs[0](data.x,data.edge_index,None)
            out.append(val)
        if self.separate_conv:
            for i in range(self.k):
                val = self.convs[i](data.x,hop_matrices[i].long(),None)
                out.append(val)
        else:
            for i in range(self.k):
                val = self.convs[0](data.x,hop_matrices[i].long(),None)
                out.append(val)
        out = torch.cat([i.unsqueeze(1) for i in out],dim=1)  # out --> N*K*H
        if isinstance(self.combine,str):
            out = torch.sum(out,dim=1)  # N*H
        else:
            out = self.combine(out)
        out = self.dropout(self.act(out))
        data.x = out
        return data


class GCNEConv(MessagePassing):
    def __init__(self, in_channels, out_channels,normed=True):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels)
        self.edge_emb = nn.Linear(in_channels,out_channels,bias=False)
        self.normed = normed
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.edge_emb.reset_parameters()


    def forward(self, x,edge_index,edge_attr=None):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_feats = None
        if edge_attr is not None:
            edge_index, edge_attr = add_self_loops(edge_index, num_nodes=x.size(0),edge_attr=edge_attr,fill_value=0.0)
            edge_feats = self.edge_emb(edge_attr)
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)


        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Step 4-5: Start propagating messages
        out = self.propagate(edge_index, x=x, norm=norm,edge_feats=edge_feats)
        if edge_attr is not None:
            return out,edge_attr
        else:
            return out

    def message(self, x_j, norm,edge_feats):
        # x_j has shape [E, out_channels]
        if edge_feats is not None:
        # Step 4: Normalize node features.
            if self.normed:
                return norm.view(-1, 1) * (x_j+edge_feats)
            else:
                return x_j + edge_feats
        else:
            if self.normed:
                return norm.view(-1, 1) * x_j
            else:
                return x_j


def new(data):
    """returns a new torch_geometric.data.Data containing the same tensors.
    Faster than data.clone() (tensors are not cloned) and preserves the old data object as long as
    tensors are not modified in-place. Intended to modify data object, without modyfing the tensors.
    ex:
    d1 = data(x=torch.eye(3))
    d2 = new(d1)
    d2.x = 2*d2.x
    In that case, d1.x was not changed."""
    return copy(data)

if __name__ == '__main__':
    d = Processed_Dataset(root='../data/MUTAG/')
    print ('num feats:',d.num_features)
    in_dim = d.num_features
    edge_dim = d.num_edge_features
    out_dim = 64
    dl = DataLoader(d,batch_size=5,shuffle=False)
    model = GCNE_Model(d,dim=32,dropout=0.5,k=4,separate_hop_conv=True,combine='add',layers = 1,feature_fusion='weighted',JK='last')

    # for d in dl:
    #     val = model(d)
    #     break
    # print (val.shape)



    # model = GCN(d, 2, 64)
    # model.reset_parameters()
    # # total_loss = 0.
    # optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # for _ in range(100):
    #     for data in dl:
    #         optimizer.zero_grad()
    #         out = model(data)
    #         print (data.y)
    #         loss = F.nll_loss(F.log_softmax(out,dim=-1), data.y.view(-1))
    #         loss.backward()
    #         optimizer.step()
    #         print (loss.item())

