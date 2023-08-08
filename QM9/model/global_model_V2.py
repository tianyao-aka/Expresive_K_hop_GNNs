from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
from torch import nn
from base_model import *
import sys
import numpy as np
sys.path.append('..')
# from nov.dataset_processing import Processed_Dataset
from torch_geometric.data import DataLoader
from dataset_processing import Processed_Dataset
from func_util import k_fold_without_validation

import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import torch.nn as nn

import torch
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import AUROC,Accuracy
from base_model import GIN,GCN,PPGN
from GCN import GCNE_Model
from GINE import GINE_Model


class GlobalModel(torch.nn.Module):
    def __init__(self,dataset,out_dim,only_base_gnn,only_mhc,use_together,Base_GNN=None,base_dropout=0.5,ks_layer='gin',mhc_dropout=0.5,base_layer=2,mhc_layer=1,mhc_num_hops=3,separate_conv = True,jk = 'concat',feature_fusion='weighted',combine='geometric',pooling_method='sum',virtual_node=False, *args, **kwargs):
        super(GlobalModel, self).__init__()
        if use_together+only_mhc+only_base_gnn>1:
            assert 'Wrong Configuration on model input!'
        self.models = nn.ModuleList()
        self.use_base_gnn = only_base_gnn
        self.use_mhc = only_mhc
        self.both = use_together
        if use_together:
            self.lin = nn.Linear(2*out_dim,1)
        else:
            self.lin = nn.Linear(out_dim, 1)
        if only_base_gnn:
            print ('use base gnn................')
            # self.gnn = Base_GNN(dataset,base_layer,out_dim,base_dropout)
            self.models.append(Base_GNN(dataset,base_layer,out_dim,base_dropout,pooling_method=pooling_method))
        if only_mhc:
            print ('use mhc............')
            if ks_layer=='gcn':
                # ##    data, hidden=64, layers=3, dropout=0.1,k=4,feature_fusion='average',JK='last',pooling_method='sum',combine='add',virtual_node=False
                self.models.append(GCNE_Model(dataset,out_dim,mhc_dropout,mhc_num_hops,separate_conv,combine,mhc_layer,feature_fusion,JK=jk))
            if ks_layer=='gin':
                self.models.append(GINE_Model(dataset,out_dim,mhc_layer,mhc_dropout,mhc_num_hops,feature_fusion,JK=jk,pooling_method=pooling_method,combine=combine,virtual_node=virtual_node))
        if use_together:
            print ('use both.......................')
            self.gnn = Base_GNN(dataset, base_layer, out_dim, base_dropout,pooling_method=pooling_method)
            if ks_layer == 'gcn':
                self.sek_model = GCNE_Model(dataset,out_dim,mhc_dropout,mhc_num_hops,separate_conv,combine,mhc_layer,feature_fusion,JK=jk)
            if ks_layer=='gin':
                # data, hidden=64, layers=3, dropout=0.5,k=4,feature_fusion='average',JK='last'
                self.sek_model = GINE_Model(dataset,out_dim,mhc_layer,mhc_dropout,mhc_num_hops,feature_fusion,JK=jk,pooling_method=pooling_method,combine=combine,virtual_node=virtual_node)
            self.models.append(self.gnn)
            self.models.append(self.sek_model)

    def reset_parameters(self):
        for conv in self.models:
            conv.reset_parameters()

    def forward(self,data):
        # input: multi-hop feature; output: node representations
        hs = []
        for conv in self.models:
            h = conv(data)
            hs.append(h)
        if self.use_base_gnn or self.use_mhc:
            h = self.lin(hs[0])
        else:
            h = self.lin(torch.cat(hs,dim=1))
        # rep = torch.sum(torch.cat([k.unsqueeze(1) for k in hs],dim=1),dim=1)
        return h
    def __repr__(self):
        return self.__class__.__name__

if __name__ == '__main__':
    d = Processed_Dataset(root='../data/MUTAG/')
    print ('num feats:',d.num_features)
    in_dim = d.num_features
    edge_dim = d.num_edge_features
    out_dim = 64
    dl = DataLoader(d,batch_size=64,shuffle=False)
    model = GlobalModel(d,out_dim,only_base_gnn=False,only_mhc=False,use_together=True,Base_GNN=GIN,base_dropout=0.5,ks_layer='gin',mhc_dropout=0.5,base_layer=2,mhc_layer=2,mhc_num_hops=4,separate_conv = True,jk = 'last',feature_fusion='weighted',combine='geometric')
    #model = GCNE_Model(d,dim=32,dropout=0.5,k=4,separate_hop_conv=True,combine='add',layers = 1,feature_fusion='weighted',JK='last')
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(100):
        for data in dl:
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(F.log_softmax(out,dim=-1), data.y.view(-1))
            loss.backward()
            optimizer.step()
            print (loss.item())



