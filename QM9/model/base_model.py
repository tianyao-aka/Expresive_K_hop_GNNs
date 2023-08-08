from torch_geometric.nn import SAGEConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINEConv,GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
import sys
sys.path.append('../')
# from nov.dataset_processing import Processed_Dataset
from dataset_processing import Processed_Dataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation
from torch import tensor
from torch.optim import Adam
import torch.nn as nn
from torch_scatter import scatter_mean,scatter_sum
from func_util import weight_reset


class QM9InputEncoder(nn.Module):
    def __init__(self, hidden_size, use_pos=False):
        super(QM9InputEncoder, self).__init__()
        self.use_pos = use_pos
        if use_pos:
            input_size = 22
        else:
            input_size = 19
        self.init_proj = nn.Linear(input_size, hidden_size)
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


class GraphSAGE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden,dropout=0.5,pooling_method='sum', *args, **kwargs):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.dropout_val = dropout
        self.reset_parameters()
        if pooling_method=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(num_layers*hidden,1))
        elif pooling_method=='sum':
            self.pool = global_add_pool
        elif pooling_method=='mean':
            self.pool = global_mean_pool


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.pool(torch.cat(xs, dim=1), batch)
        x = F.relu(self.lin1(x))
        if self.dropout_val>0:
            x = self.dropout(x)
        # x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


class GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden,dropout=0.5,pooling_method='sum', *args, **kwargs):
        super(GIN, self).__init__()
        edge_dim=None
        self.init_proj = QM9InputEncoder(hidden_size=hidden)
        if 'edge_attr' in dataset[0]:
            edge_dim = dataset[0].edge_attr.shape[1]
            self.edge_emb = nn.Linear(edge_dim,hidden,bias=False)
            self.conv1 = GINEConv(
                Sequential(
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU()
                ),
                train_eps=False,edge_dim=hidden)
        else:
            self.conv1 = GINConv(
                Sequential(
                    Linear(dataset.num_features, hidden),
                    BN(hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    BN(hidden),
                    ReLU(),
                ),
                train_eps=False)
        self.convs = torch.nn.ModuleList()
        self.dropout_val = dropout
        for i in range(num_layers - 1):
            if 'edge_attr' in dataset[0]:
                self.convs.append(
                    GINEConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=False,edge_dim=hidden))
            else:
                self.convs.append(
                    GINConv(
                        Sequential(
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU(),
                            Linear(hidden, hidden),
                            BN(hidden),
                            ReLU()
                        ),
                        train_eps=False))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        if pooling_method=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(num_layers*hidden,1))
        elif pooling_method=='sum':
            self.pool = global_add_pool
        elif pooling_method=='mean':
            self.pool = global_mean_pool
        # self.lin2 = Linear(hidden, dataset.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.apply(weight_reset)


    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        edge_index, batch = data.edge_index, data.batch
        x = self.init_proj(data)
        if 'edge_attr' in data:
            e = data.edge_attr
            e = self.edge_emb(e)
            x = self.conv1(x, edge_index,e)
        else:
            x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            if 'edge_attr' in data:
                x = conv(x, edge_index,e)
            else:
                x = conv(x, edge_index)
            xs += [x]

        x = self.pool(torch.cat(xs, dim=1), batch)
        x = F.relu(self.lin1(x))
        if self.dropout_val>0:
            x = self.dropout(x)
        # x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__



class GCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden,dropout=0.5,pooling_method='sum', *args, **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(num_layers * hidden, hidden)
        self.dropout_val = dropout
        self.dropout = nn.Dropout(dropout)
        if pooling_method=='attention':
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(num_layers*hidden,1))
        elif pooling_method=='sum':
            self.pool = global_add_pool
        elif pooling_method=='mean':
            self.pool = global_mean_pool
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.pool(torch.cat(xs, dim=1), batch)
        # x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        if self.dropout_val>0:
            x = self.dropout(x)
        return x


    def __repr__(self):
        return self.__class__.__name__


# class GNN(nn.Module):
#     def __init__(self,data,hidden_dim,layer_name='gcn',layer_num = 3):
#         super().__init__()
#         self.name = layer_name
#         self.layer_num = layer_num
#         self.relu1 = nn.ReLU()
#         self.relu2 = nn.ReLU()
#         self.relu3 = nn.ReLU()
#         self.relu4 = nn.ReLU()
#         self.nn_modules = nn.ModuleList()
#         dim = data.num_features
#         if 'edge_attr' in data[0]:
#             edge_dim = data[0].edge_attr.shape[1]
#             self.edge_emb = nn.Linear(edge_dim,hidden_dim)
#         if layer_name=='gcn':
#             self.gcn1 = GCNConv(dim, hidden_dim)
#             self.gcn2 = GCNConv(hidden_dim, hidden_dim)
#             if layer_num>2:
#                 for _ in range(layer_num-2):
#                     self.nn_modules.append(GCNConv(hidden_dim, hidden_dim))
#                     self.nn_modules.append(nn.ReLU())
#
#         elif layer_name=='sage':
#             self.gcn1 = SAGEConv(dim, hidden_dim)
#             self.gcn2 = SAGEConv(hidden_dim, hidden_dim)
#             if layer_num > 2:
#                 for _ in range(layer_num - 2):
#                     self.nn_modules.append(SAGEConv(hidden_dim, hidden_dim))
#                     self.nn_modules.append(nn.ReLU())
#         elif layer_name=='gin':
#             nn_callable1 = nn.Sequential(nn.Linear(dim,hidden_dim),nn.BatchNorm1d(num_features=hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(num_features=hidden_dim))
#             nn_callable2 = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(num_features=hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim),nn.BatchNorm1d(num_features=hidden_dim))
#             nn_callable3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(num_features=hidden_dim),nn.ReLU(),
#                                          nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(num_features=hidden_dim))
#             nn_callable4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),nn.BatchNorm1d(num_features=hidden_dim), nn.ReLU(),
#                                          nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(num_features=hidden_dim))
#             self.gin1 = GINEConv(nn=nn_callable1,edge_dim=hidden_dim)
#             self.gin2 = GINEConv(nn=nn_callable2,edge_dim=hidden_dim)
#             self.gin3 = GINEConv(nn=nn_callable3,edge_dim=hidden_dim)
#             self.gin4 = GINEConv(nn=nn_callable4,edge_dim=hidden_dim)
#             self.nn_modules.append(self.gin3)
#             self.nn_modules.append(self.gin4)
#         else:
#             print ('gnn module error')
#
#
#     def forward(self,data):
#         x,edges,batch = data.x,data.edge_index,data.batch
#         if self.name=='gin':
#             if 'edge_attr' in data:
#                 e = self.edge_emb(data.edge_attr)
#                 h = self.gin1(x,edges,e)
#                 h = self.gin2(h,edges,e)
#                 for func in self.nn_modules:
#                     h = func(h,edges,e)
#                 return h
#             else:
#                 h = self.gin1(x,edges)
#                 h = self.gin2(h,edges)
#                 for func in self.nn_modules:
#                     h = func(h,edges)
#                 return h
#         else:
#             h = self.relu1(self.gcn1(x,edges))
#             h = self.relu2(self.gcn2(h,edges))
#             for func in self.nn_modules:
#                 if isinstance(func,nn.ReLU):
#                     h = func(h)
#                 else:
#                     h = func(h,edges)
#             h  = global_add_pool(h, batch)
#             return h


class PPGN(torch.nn.Module):
    def __init__(self,ninp=3, nmax=30, nneuron=40,hidden2=32):
        super(PPGN, self).__init__()

        self.nmax = nmax
        self.nneuron = nneuron

        bias = False
        self.mlp1_1 = torch.nn.Conv2d(ninp, nneuron, 1, bias=bias)
        self.mlp1_2 = torch.nn.Conv2d(ninp, nneuron, 1, bias=bias)
        self.mlp1_3 = torch.nn.Conv2d(nneuron + ninp, nneuron, 1, bias=bias)

        self.mlp2_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp2_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp2_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.mlp3_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp3_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp3_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.mlp4_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp4_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp4_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.mlp5_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp5_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp5_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.h1 = torch.nn.Linear(2 * 5 * nneuron, hidden2)
        self.h2 = torch.nn.Linear(hidden2, 1)

    def forward(self, data):
        x = data.X2
        M = torch.sum(data.M, (1), True)
        x1 = F.relu(self.mlp1_1(x) * M)
        x2 = F.relu(self.mlp1_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp1_3(torch.cat([x1x2, x], 1)) * M)

        # sum or mean layer readout
        xo1 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)
        # xo1=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x1 = F.relu(self.mlp2_1(x) * M)
        x2 = F.relu(self.mlp2_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp2_3(torch.cat([x1x2, x], 1)) * M)

        # sum or mean layer readout
        xo2 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)
        # xo2=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x1 = F.relu(self.mlp3_1(x) * M)
        x2 = F.relu(self.mlp3_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp3_3(torch.cat([x1x2, x], 1)) * M)

        # sum or mean layer readout
        xo3 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)
        # xo3=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x1 = F.relu(self.mlp4_1(x) * M)
        x2 = F.relu(self.mlp4_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp4_3(torch.cat([x1x2, x], 1)) * M)

        # sum or mean layer readout
        xo4 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)
        # xo4=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x1 = F.relu(self.mlp5_1(x) * M)
        x2 = F.relu(self.mlp5_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp5_3(torch.cat([x1x2, x], 1)) * M)

        # sum or mean layer readout
        xo5 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)
        # xo3=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x = torch.cat([xo1, xo2, xo3, xo4, xo5], 1)
        x = F.relu(self.h1(x))
        return x


if __name__ == '__main__':
    d = Processed_Dataset(root='../data/MUTAG/')
    print ('num feats:',d.num_features)
    dl = DataLoader(d,batch_size=6)

    model = GIN(d,3,64)
    for x in dl:
        h=model(x)
        print (h.shape)
        break

    # total_loss = 0.
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



