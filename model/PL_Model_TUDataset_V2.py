from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
from torch import nn
from base_model import *
from global_model_V2 import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import AUROC,Accuracy

# self,dataset,out_dim,only_base_gnn,only_mhc,use_together,Base_GNN=None,base_dropout=0.5,ks_layer='gcn',mhc_dropout=0.5,base_layer=2,mhc_layer=1,mhc_num_hops=3,separate_conv = True,jk = 'concat',feature_fusion='weighted',combine='geometric', *args, **kwargs):

class PL_TU_Model(pl.LightningModule):
    def __init__(self, dataset,out_dim,only_base_gnn,only_mhc,use_together,base_gnn_str,base_dropout=0.5,mhc_layer='gcn',mhc_dropout=0.5,base_layer=2,mhc_num_layers=1,mhc_num_hops=3,separate_conv = True,jk = 'concat',feature_fusion='weighted',combine='geometric',lr=1e-3,weight_decay=1e-5, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.acc = Accuracy(top_k=1)
        self.lr= lr
        print ('learning rate',self.lr)
        self.weight_decay = weight_decay
        self.global_model = GlobalModel(dataset,out_dim,only_base_gnn,only_mhc,use_together,Base_GNN=self.get_base_gnn_model(base_gnn_str),base_dropout=base_dropout,ks_layer=mhc_layer,mhc_dropout=mhc_dropout,base_layer=base_layer,mhc_layer=mhc_num_layers,mhc_num_hops=mhc_num_hops,separate_conv = separate_conv,jk = jk,feature_fusion=feature_fusion,combine=combine)
        self.val_acc = []
        self.record_acc = []

    def get_base_gnn_model(self,base_gnn_str):
        if base_gnn_str=='GCN':
            return GCN
        if base_gnn_str=='GIN':
            return  GIN
        if base_gnn_str=='SAGE':
            return GraphSAGE

    def forward(self,data):
        pass

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
        return {'optimizer':optimizer,'lr_scheduler':scheduler}

    def training_step(self, batch, batch_idx):
        h = self.global_model(batch)
        h = F.log_softmax(h,dim=-1)
        y = batch.y.view(-1,)
        loss_val = F.nll_loss(h, y)
        self.log("train_loss_tudataset", loss_val.item(), prog_bar=True, logger=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        h = self.global_model(batch)
        h = F.log_softmax(h,dim=-1)
        y = batch.y.view(-1,)
        loss_val = F.nll_loss(h, y)
        acc = self.acc(h,y)
        metrics = {'val_acc':acc,'val_loss':loss_val}
        pred = torch.argmax(h, dim=1)
        tot = len(y)
        correct = torch.sum(y == pred).item()
        self.val_acc.append((tot, correct))

        self.log_dict(metrics,prog_bar=True,logger=True)


    def on_validation_epoch_end(self) -> None:
        tot = sum([i[0] for i in self.val_acc])
        correct = sum([i[1] for i in self.val_acc])
        val = 1.*correct/tot
        self.record_acc.append(val)
        self.val_acc = []

    def test_step(self, batch, batch_idx):
        h = self.forward(batch)
        y = batch.y
        loss_val = F.nll_loss(h, y)
        acc = self.acc(h,y)
        metrics = {'val_acc':acc,'val_loss':loss_val}
        self.log_dict(metrics,prog_bar=True,logger=True)


if __name__ =='__main__':
    d = Processed_Dataset(root='../data/MUTAG/')
    print ('num feats:',d.num_features)
    in_dim = d.num_features
    edge_dim = d.num_edge_features
    out_dim = 64
    dl = DataLoader(d,batch_size=64,shuffle=False)

