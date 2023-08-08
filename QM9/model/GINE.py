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

class QM9InputEncoder(nn.Module):
    def __init__(self, hidden_size,rw_size, use_pos=False):
        super(QM9InputEncoder, self).__init__()
        self.use_pos = use_pos
        if use_pos:
            input_size = 22
        else:
            input_size = 19
        self.init_proj = nn.Linear(input_size+rw_size, hidden_size)
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


class GINE_Model(nn.Module):
    def __init__(self,data, hidden=64, layers=3, dropout=0.,k=4,feature_fusion='average',JK='last',pooling_method='sum',combine='add',virtual_node=False):
        super().__init__()
        self.feature_fusion = feature_fusion
        in_dim = data.num_features + data[0].rw_feature.shape[1]
        if 'edge_attr' in data[0]:
            edge_dim = data[0].edge_attr.shape[1]
            self.edge_emb = nn.Linear(edge_dim, hidden,bias=False)
        self.k = k
        self.JK = JK
        self.node_emb = QM9InputEncoder(hidden_size=hidden,rw_size=data[0].rw_feature.shape[1])
        convs = [ConvBlock(hidden,
                           dropout=dropout,
                           k=min(i + 1, k),combine=combine,virtual_node=virtual_node)
                 for i in range(layers - 1)]
        convs.append(ConvBlock(hidden,
                               dropout=dropout,
                               last_layer=True,
                               k=min(layers, k),combine=combine,virtual_node=virtual_node))
        self.model = nn.Sequential(*convs)
        self.alphas = nn.Parameter(torch.zeros(1, 3, data[0].rw_feature.shape[1]))
        if self.JK == 'concat':
            self.JK_layer = nn.Sequential(nn.Linear((layers + 1) * hidden, hidden),nn.ReLU())
        elif self.JK=='attention':
            self.JK_layer = nn.LSTM(hidden, layers, 1, batch_first=True, bidirectional=True,dropout=dropout)
            self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())

        if pooling_method=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden,1))
        elif pooling_method=='sum':
            self.pool = global_add_pool
        elif pooling_method=='mean':
            self.pool = global_mean_pool
        if virtual_node:
            self.virtualnode_embedding = torch.nn.Embedding(1, hidden)
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
        self.virtual_node = virtual_node
        self.apply(weight_reset)

    def forward(self, data):

        if self.virtual_node:
            torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)
            virtualnode_embedding = self.virtualnode_embedding(torch.zeros(data.batch[-1].item() + 1).to(data.edge_index.dtype).to(data.edge_index.device))
            data.virtualnode_embedding = virtualnode_embedding

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
        data.x = x
        h = self.node_emb(data)
        data.x = [h]
        ptr = data.ptr
        hop1,hop2,hop3,hop4,hop5 = data.hop1,data.hop2,data.hop3,data.hop4,data.hop5
        hop1 = collate_edge_index(hop1,ptr,use_cuda)
        hop2 = collate_edge_index(hop2, ptr, use_cuda)
        hop3 = collate_edge_index(hop3, ptr, use_cuda)
        hop4 = collate_edge_index(hop4, ptr, use_cuda)
        hop5 = collate_edge_index(hop5, ptr, use_cuda)
        # hop6 = collate_edge_index(hop6, ptr, use_cuda)
        data.collated_hop_matrices = [hop1,hop2,hop3,hop4,hop5]
        g = self.model(data)
        if self.JK =='last':
            h = g.x[0]
        elif self.JK=='concat':
            h = self.JK_layer(torch.cat(g.x,dim=1))
        elif self.JK=='attention':
            self.JK_layer.flatten_parameters()
            h_list = [h.unsqueeze(0) for h in g.x]
            h_list = torch.cat(h_list, dim=0).transpose(0, 1)  # N *num_layer * H
            attention_score, _ = self.JK_layer(h_list)  # N * num_layer * 2*num_layer
            attention_score = torch.softmax(torch.sum(attention_score, dim=-1), dim=1).unsqueeze(-1)
            h = self.proj(torch.sum(h_list * attention_score, dim=1))
        return self.pool(h,data.batch)


class ConvBlock(nn.Module):
    def __init__(self, dim,dropout=0.1, activation=F.gelu, k=4, last_layer=False,combine='add',virtual_node = False):
        super().__init__()
        self.edge_emb = nn.Linear(dim,dim,bias=False)
        self.conv = GINEPLUS(dim, k=k,combine=combine)
        self.norm = nn.BatchNorm1d(dim)
        self.act = activation or nn.Identity()
        self.last_layer = last_layer
        self.dropout = dropout
        self.virtual_node = virtual_node
        if virtual_node:
            self.mlp_virtualnode = torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.BatchNorm1d(dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(dim, dim), torch.nn.BatchNorm1d(dim), torch.nn.ReLU())

    def forward(self, data):
        data = new(data)
        if 'edge_attr' in data:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            edge_attr = None
        hop_matrices = data.collated_hop_matrices
        h = x
        if self.virtual_node:
            virtualnode_embedding = data.virtualnode_embedding
            H = self.conv(h,hop_matrices , edge_attr,virtualnode_embedding,batch)
        else:
            H = self.conv(h, hop_matrices, edge_attr, None, batch)
        h = H[0]
        if not self.last_layer:
            h = self.act(h)
        h = self.norm(h)
        h = F.dropout(h, self.dropout, training=self.training)
        if self.virtual_node:
            virtualnode_embedding_temp = global_add_pool(h,batch) + virtualnode_embedding
            virtualnode_embedding = F.dropout(self.mlp_virtualnode(virtualnode_embedding_temp),self.dropout, training=self.training)
            data.virtualnode_embedding = virtualnode_embedding
        H[0] = h
        h = H
        data.x = h
        return data


class GINEPLUS(MessagePassing):
    def __init__(self, dim, k=4,combine='add',virtual_node =False,dropout = 0.5, **kwargs):
        super().__init__(aggr='add')
        self.k = k
        if combine=='geometric':
            self.combine = GeometricCombine(k+1,dim)
        else:
            self.combine = 'add'
        self.nn=nn.Sequential(nn.Linear(dim,dim),
                                      nn.BatchNorm1d(dim),
                                      nn.ReLU(),
                                      nn.Linear(dim,dim),
                                      nn.BatchNorm1d(dim),
                                      nn.ReLU())
        self.eps = nn.Parameter(torch.zeros(k + 1, dim), requires_grad=True)

        self.virtual_node = virtual_node

    def forward(self, XX, hop_matrix_list, edge_attr,virtualnode_embedding=None,batch=None):
        """Warning, XX is now a list of previous xs, with x[0] being the last layer"""
        assert len(XX) >= self.k
        if edge_attr is not None:
            assert XX[-1].size(-1) == edge_attr.size(-1)
        # result = (1 + self.eps[0]) * XX[0]
        if virtualnode_embedding is None:
            result = XX[0]
        else:
            result = XX[0] + virtualnode_embedding[batch]
        outputs = [result]
        for i, x in enumerate(XX):
            if i >= self.k:
                break
            if i == 0:
                out = self.propagate(hop_matrix_list[i], edge_attr=edge_attr, x=x)
            else:
                out = self.propagate(hop_matrix_list[i], edge_attr=None, x=x)
            # result += (1 + self.eps[i + 1]) * out
            outputs.append(out)
        outputs = torch.cat([i.unsqueeze(1) for i in outputs], dim=1)  # out --> N*K*H
        if isinstance(self.combine, str):
            result = torch.sum(outputs, dim=1)  # N*H
        else:
            result = self.combine(outputs)
        result = self.nn(result)
        return [result] + XX

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return x_j + edge_attr
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

