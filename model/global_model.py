from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
from torch import nn
from base_model import *
from m2hc import *
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import AUROC,Accuracy



class GlobalModel(torch.nn.Module):
    def __init__(self,dataset,in_dim, out_dim,only_base_gnn,only_mhc,use_together,Base_GNN=None,base_dropout=0.5,mhc_dropout=0.5,base_layer=2,mhc_layer=1,mhc_num_hops=3, *args, **kwargs):
        super(GlobalModel, self).__init__()
        if use_together+only_mhc+only_base_gnn>1:
            assert 'Wrong Configuration on model input!'
        self.models = nn.ModuleList()
        self.use_base_gnn = only_base_gnn
        self.use_mhc = only_mhc
        self.both = use_together
        if use_together:
            self.lin = nn.Linear(2*out_dim,dataset.num_classes)
        else:
            self.lin = nn.Linear(out_dim, dataset.num_classes)
        if only_base_gnn:
            print ('use base gnn................')
            # self.gnn = Base_GNN(dataset,base_layer,out_dim,base_dropout)
            self.models.append(Base_GNN(dataset,base_layer,out_dim,base_dropout))
        if only_mhc:
            print ('use mhc............')
            # self.mhc_model = MHC_GNN(in_dim,out_dim,mhc_dropout,num_layers=mhc_layer,num_hops=mhc_num_hops)
            self.models.append(MHC_GNN(in_dim,out_dim,mhc_dropout,num_layers=mhc_layer,num_hops=mhc_num_hops))
        if use_together:
            print ('use both.......................')
            self.gnn = Base_GNN(dataset, base_layer, out_dim, base_dropout)
            self.mhc_model = MHC_GNN(in_dim, out_dim, mhc_dropout, num_layers=mhc_layer, num_hops=mhc_num_hops)
            self.models.append(self.gnn)
            self.models.append(self.mhc_model)

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
    dl = DataLoader(d,batch_size=64)
    # # dataset,in_dim, out_dim,only_base_gnn,only_mhc,use_together,Base_GNN=None,base_dropout=0.5,mhc_dropout=0.5,base_layer=2,mhc_layer=1,mhc_num_hops=3
    # params = {'dataset':d,'in_dim':d.num_features,'only_base_gnn':True,'only_mhc':False,'use_together':False,'out_dim':64,'Base_GNN':GCN}
    # model = GlobalModel(**params)
    # pmodel = PL_TU_Model(model)
    # trainer = pl.Trainer(default_root_dir=f'saved_model/45345',
    #                      accelerator='cpu', max_epochs=100)
    # trainer.fit(model=pmodel, train_dataloaders=dl, val_dataloaders=dl)
    # optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # for _ in range(100):
    #     for data in dl:
    #         optimizer.zero_grad()
    #         out = model(data)
    #         loss = F.nll_loss(F.log_softmax(out,dim=-1), data.y.view(-1))
    #         loss.backward()
    #         optimizer.step()
    #         print (loss.item())



