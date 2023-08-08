from torch_geometric.nn import SAGEConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
from torch import nn
import sys
sys.path.append('..')
from dataset_processing import Processed_Dataset
from torch_geometric.data import DataLoader


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


class MHC_GNN_Layer(torch.nn.Module):
    def __init__(self,in_dim, out_dim,dropout=0.5,num_hops=3, *args, **kwargs):
        super(MHC_GNN_Layer, self).__init__()
        self.models = nn.ModuleList()
        self.dropout=dropout
        for _ in range(num_hops+1):
            self.models.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
                      nn.Linear(out_dim, out_dim)))

    def reset_parameters(self):
        for conv in self.models:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
            if isinstance(conv,nn.Sequential):
                for sub_conv in conv:
                    if hasattr(conv, 'reset_parameters'):
                        sub_conv.reset_parameters()

    def forward(self, *node_hop_features):
        # input: multi-hop feature; output: node representations
        xs = []
        for idx,conv in enumerate(self.models):
            x = self.models[idx](node_hop_features[idx])
            xs += [x]
        x = torch.sum(torch.cat([h.unsqueeze(1) for h in xs],dim=1),dim=1)
        return x

    def __repr__(self):
        return self.__class__.__name__

class MHC_GNN(torch.nn.Module):
    def __init__(self,in_dim, out_dim,dropout=0.5,num_layers=1,num_hops=3,concat=True, *args, **kwargs):
        super(MHC_GNN, self).__init__()
        self.models = nn.ModuleList()
        self.concat = concat
        self.dropout=dropout
        self.num_hops = num_hops
        self.num_layers = num_layers
        self.first_layer = MHC_GNN_Layer(in_dim,out_dim,dropout,num_hops)
        self.models = nn.ModuleList()
        if num_layers==1:
            self.concat = False
        if self.concat:
            self.lin = nn.Linear(num_layers*out_dim,out_dim)
        for i in range(num_layers-1):
            self.models.append(MHC_GNN_Layer(out_dim,out_dim,dropout,num_hops))

    def forward(self, data):
        batch = data.batch
        f = torch.cat((data.x,data.rw_feature),dim=1)
        hop_mat = [data[f'hop{int(i)}'] for i in range(1,self.num_hops+1)]
        h = [data[f'hop{int(i)}_features'] for i in range(1,self.num_hops+1)]
        h = [f]+h
        x = self.first_layer(*h)
        out=[x]
        for j in range(self.num_layers-1):
            hs = [x]
            for i in range(self.num_hops):
                hop_matrix = collate_graph_adj(hop_mat[i],data.ptr,use_gpu=True if torch.cuda.is_available() else False)
                feats = torch.sparse.mm(hop_matrix, x)
                hs.append(feats)
            x = self.models[j](*hs)
            out.append(x)

        if self.concat:
            rep = global_add_pool(self.lin(torch.cat(out, dim=1)),batch)
        else:
            h = torch.cat([i.unsqueeze(1) for i in out],dim=1)
            rep = global_add_pool(torch.sum(h,dim=1),batch)
        return rep

    def reset_parameters(self):
        self.first_layer.reset_parameters()
        for conv in self.models:
            conv.reset_parameters()


if __name__ == '__main__':
    d = Processed_Dataset(root='../data/MUTAG/')
    # dl = DataLoader(d,batch_size=2,shuffle=False)
    nn.Linear(2,1).reset_parameters()
    model = MHC_GNN(d.num_features,64,num_hops=3,num_layers=2)
    model.reset_parameters()

    # for d in dl:
    #     d.hop1_features = d.x
    #     d.hop2_features = d.x
    #     d.hop3_features = d.x
    #     out = model(d)
    #     print (out.shape)
    #     break


