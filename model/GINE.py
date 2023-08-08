import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,global_add_pool,global_mean_pool
from combine import *
import math
import sys
sys.path.append('..')
# from nov.dataset_processing import Processed_Dataset
from dataset_processing import Processed_Dataset
from torch_geometric.data import DataLoader
from func_util import collate_graph_adj,collate_edge_index,weight_reset
from copy import copy
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation


use_cuda =True if torch.cuda.is_available() else False
gpu_num = 1 if torch.cuda.is_available() else None


class GINE_Model(nn.Module):
    def __init__(self,data, hidden=64, layers=3, dropout=0.5,k=4,feature_fusion='average',JK='last',pooling_method='sum'):
        super().__init__()
        self.feature_fusion = feature_fusion
        in_dim = data.num_features + data[0].rw_feature.shape[1]
        if 'edge_attr' in data[0]:
            edge_dim = data[0].edge_attr.shape[1]
            self.edge_emb = nn.Linear(edge_dim, hidden)
        self.k = k
        self.JK = JK
        self.node_emb = nn.Linear(in_dim,hidden)
        convs = [ConvBlock(hidden,
                           dropout=dropout,
                           k=min(i + 1, k))
                 for i in range(layers - 1)]
        convs.append(ConvBlock(hidden,
                               dropout=dropout,
                               last_layer=True,
                               k=min(layers, k)))
        self.model = nn.Sequential(*convs)
        self.alphas = nn.Parameter(torch.zeros(1, 3, data[0].rw_feature.shape[1]))
        if self.JK == 'concat':
            self.JK_layer = nn.Linear((layers+1) * hidden, hidden)
        if pooling_method=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden,1))
        elif pooling_method=='sum':
            self.pool = global_add_pool
        elif pooling_method=='mean':
            self.pool = global_mean_pool
        self.apply(weight_reset)

    def forward(self, data):
        if 'edge_attr' in data:
            edge_attr = self.edge_emb(data.edge_attr)
            data.edge_attr = edge_attr
        else:
            data.edge_attr = None
        x,rw,central_to_subgraph,samehop = data.x,data.rw_feature,data.central_to_subgraph_feats,data.context_samehop_feats
        if self.feature_fusion == 'average':
            avg_feats = (rw+central_to_subgraph+samehop)/3.0
        else:
            f = torch.cat([rw.unsqueeze(1),central_to_subgraph.unsqueeze(1),samehop.unsqueeze(1)],dim=1)   # N*3*F
            w = F.softmax(self.alphas,dim=1)
            avg_feats = torch.sum(f*w,dim=1)   # N*F
        x = torch.cat([x,avg_feats],dim=1)
        h = self.node_emb(x)
        data.x = [h]
        ptr = data.ptr
        hop1,hop2,hop3,hop4,hop5 = data.hop1,data.hop2,data.hop3,data.hop4,data.hop5
        hop1 = collate_edge_index(hop1,ptr,use_cuda)
        hop2 = collate_edge_index(hop2, ptr, use_cuda)
        hop3 = collate_edge_index(hop3, ptr, use_cuda)
        hop4 = collate_edge_index(hop4, ptr, use_cuda)
        hop5 = collate_edge_index(hop5, ptr, use_cuda)
        data.collated_hop_matrices = [hop1,hop2,hop3,hop4,hop5]
        g = self.model(data)
        if self.JK =='last':
            h = g.x[0]
        else:
            h = self.JK_layer(torch.cat(g.x,dim=1))
        return self.pool(h,data.batch)



class ConvBlock(nn.Module):
    def __init__(self, dim,dropout=0.5, activation=F.relu, k=4, last_layer=False):
        super().__init__()
        self.edge_emb = nn.Linear(dim,dim,bias=False)
        self.conv = GINEPLUS(dim, k=k)
        self.norm = nn.BatchNorm1d(dim)
        self.act = activation or nn.Identity()
        self.last_layer = last_layer
        self.dropout_ratio = dropout


    def forward(self, data):
        data = new(data)
        if 'edge_attr' in data:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            edge_attr = None
        hop_matrices = data.collated_hop_matrices
        h = x
        H = self.conv(h,hop_matrices , edge_attr)
        h = H[0]
        h = self.norm(h)
        if not self.last_layer:
            h = self.act(h)
        h = F.dropout(h, self.dropout_ratio, training=self.training)
        H[0] = h
        h = H
        data.x = h
        return data


class GINEPLUS(MessagePassing):
    def __init__(self, dim, k=4, **kwargs):
        super().__init__(aggr='add')
        self.k = k
        self.nn=nn.Sequential(nn.Linear(dim,2*dim),
                                      nn.BatchNorm1d(2*dim),
                                      nn.ReLU(),
                                      nn.Linear(2*dim,dim),
                                      nn.BatchNorm1d(dim),
                                      nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(k + 1, dim), requires_grad=True)

    def forward(self, XX, hop_matrix_list, edge_attr):
        """Warning, XX is now a list of previous xs, with x[0] being the last layer"""
        assert len(XX) >= self.k
        if edge_attr is not None:
            assert XX[-1].size(-1) == edge_attr.size(-1)
        result = (1 + self.eps[0]) * XX[0]
        for i, x in enumerate(XX):
            if i >= self.k:
                break
            if i == 0:
                out = self.propagate(hop_matrix_list[i], edge_attr=edge_attr, x=x)
            else:
                out = self.propagate(hop_matrix_list[i], edge_attr=None, x=x)
            result += (1 + self.eps[i + 1]) * out
        result = self.nn(result)
        return [result] + XX

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu(x_j)

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

if __name__ =='__main__':
    d = Processed_Dataset(root='../data/MUTAG/')
    print ('num feats:',d.num_features)
    in_dim = d.num_features
    edge_dim = d.num_edge_features
    out_dim = 64
    dl = DataLoader(d,batch_size=5,shuffle=False)
    model = GINE_Model(d,64,layers=3,k=4,JK='concat',feature_fusion='weighted')
    for d in dl:
        h = model(d)
        break
    print (h.shape)


