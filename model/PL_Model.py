from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
from torch import nn
from base_model import *
from global_model import *
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


class PL_Model(pl.LightningModule):
    def __init__(self,num_classes,classification,dataset,in_dim, out_dim,only_base_gnn,only_mhc,use_together,base_gnn,base_dropout=0.5,mhc_dropout=0.5,base_layer=2,mhc_layer=1,mhc_num_hops=3,lr=5e-3,weight_decay=1e-5, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.eval_roc = AUROC(num_classes=num_classes)
        self.acc = Accuracy(top_k=1)
        self.lr = lr
        self.is_classification = classification
        self.weight_decay = weight_decay
        self.global_model = GlobalModel(dataset,in_dim,out_dim,only_base_gnn,only_mhc,use_together,self.get_base_gnn_model(base_gnn),base_dropout,mhc_dropout,base_layer,mhc_layer,mhc_num_hops)
        self.lin = nn.Linear(out_dim,num_classes)
        self.l1_loss = nn.L1Loss()

    def get_base_gnn_model(self,base_gnn_str):
        if base_gnn_str=='GCN':
            return GCN
        if base_gnn_str=='GIN':
            return  GIN
        if base_gnn_str=='SAGE':
            return GraphSAGE

    def forward(self,data):
        h = self.global_model(data)
        h = self.lin(h)
        if self.is_classification:
            h = F.log_softmax(h,dim=-1)
        else:
            # regression
            h = h.view(-1,)
        return h

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)
        return {'optimizer':optimizer,'lr_scheduler':scheduler}

    def training_step(self, batch, batch_idx):
        h = self.forward(batch)
        y = batch.y
        if self.is_classification:
            loss_val = F.nll_loss(h, y)
        else:
            loss_val = self.l1_loss(h,y)
        self.log("train_loss", loss_val.item(), prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        h = self.forward(batch)
        y = batch.y
        if self.is_classification:
            loss_val = F.nll_loss(h, y)
            acc = self.acc(h,y)
            metrics = {'val_acc':acc,'val_loss':loss_val}
        else:
            loss_val = self.l1_loss(h,y)
            metrics = { 'val_loss': loss_val}
        self.log_dict(metrics,prog_bar=True,logger=True)

    def test_step(self, batch, batch_idx):
        h = self.forward(batch)
        y = batch.y
        if self.is_classification:
            loss_val = F.nll_loss(h, y)
            acc = self.acc(h,y)
            metrics = {'val_acc':acc,'val_loss':loss_val}
        else:
            loss_val = self.l1_loss(h,y)
            metrics = { 'val_loss': loss_val}
        self.log_dict(metrics,prog_bar=True,logger=True)

# if __name__ =='__main__':
#     PL_Model()
