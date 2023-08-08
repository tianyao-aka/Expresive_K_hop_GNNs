from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
from torch import nn
import sys
sys.path.append('..')
from model.base_model import *
from model.global_model_V2 import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import AUROC,Accuracy
from QM9Dataset import QM9
from termcolor import colored

# self,dataset,out_dim,only_base_gnn,only_mhc,use_together,Base_GNN=None,base_dropout=0.5,ks_layer='gcn',mhc_dropout=0.5,base_layer=2,mhc_layer=1,mhc_num_hops=3,separate_conv = True,jk = 'concat',feature_fusion='weighted',combine='geometric', *args, **kwargs):


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414



conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

class Model(pl.LightningModule):
    def __init__(self,dataset, out_dim,only_base_gnn,only_mhc,use_together,base_gnn_str,base_dropout=0.5,mhc_layer='gcn',mhc_dropout=0.5,base_layer=2,mhc_num_layers=1,mhc_num_hops=3,separate_conv = True,jk = 'concat',feature_fusion='weighted',combine='geometric',lr=1e-3,weight_decay=1e-6,virtual_node=False,task_id=0,pooling_method='sum',test_loader=None,std=None, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.config_str = f'dim:{out_dim},layer:{mhc_num_layers},hop:{mhc_num_hops},feature:{feature_fusion},combine:{combine}'
        print (self.config_str)
        # load dataset
        # dataset = QM9('data/QM9_processed/')
        self.test_loader = test_loader
        tenpercent = int(len(dataset) * 0.1)
        self.std = std
        self.lr= lr
        self.weight_decay = weight_decay
        self.global_model = GlobalModel(dataset,out_dim,only_base_gnn,only_mhc,use_together,Base_GNN=self.get_base_gnn_model(base_gnn_str),base_dropout=base_dropout,ks_layer=mhc_layer,mhc_dropout=mhc_dropout,base_layer=base_layer,mhc_layer=mhc_num_layers,mhc_num_hops=mhc_num_hops,separate_conv = separate_conv,jk = jk,feature_fusion=feature_fusion,combine=combine,virtual_node=virtual_node,pooling_method=pooling_method)
        # d,out_dim,only_base_gnn=False,only_mhc=False,use_together=True,Base_GNN=GIN,base_dropout=0.5,ks_layer='gin',mhc_dropout=0.5,base_layer=2,mhc_layer=2,mhc_num_hops=4,separate_conv = True,jk = 'last',feature_fusion='weighted',combine='geometric'
        self.task_id = task_id
        self.best_val = 1e8
        self.test_mae = 1e8
        self.val_mae = []


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
        # We will reduce
        # the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.75,patience=15,min_lr=1e-5)
        return {'optimizer':optimizer,'lr_scheduler':scheduler,'monitor':'val_loss'}

    def training_step(self, batch, batch_idx):
        h = self.global_model(batch)
        h = h.view(-1,).float()
        y = batch.y[:,self.task_id].view(-1,).float()
        loss_val = F.mse_loss(h,y)
        # loss_val = F.smooth_l1_loss(h, y)
        self.log("train_loss", loss_val.item(), prog_bar=True, logger=False,batch_size=128)
        return loss_val



    def validation_step(self, batch, batch_idx):
        h = self.global_model(batch).view(-1, ).float()
        y = batch.y[:, self.task_id].float()
        std_val = self.std[self.task_id].to(self.device)
        

        mae_error = (h*std_val-y*std_val).abs().sum().item()
        self.val_mae.append([mae_error,batch.num_graphs])
        mae_error = mae_error/batch.num_graphs
        self.log("val_loss", mae_error, prog_bar=True, logger=False,batch_size=128)


    def on_validation_epoch_end(self) -> None:
        valid_error = sum([i[0] for i in self.val_mae])
        graphs = sum([i[1] for i in self.val_mae])
        val_mae = valid_error/graphs
        
        if val_mae<self.best_val:
            self.best_val = val_mae
            error = 0.
            num_graphs = 0.
            with torch.no_grad():
                self.global_model.eval()
                for batch in self.test_loader:
                    batch = batch.to(self.device)
                    num_graphs += batch.num_graphs
                    h = self.global_model(batch).view(-1, ).float()
                    y = batch.y[:, self.task_id].float()
                    std_val = self.std[self.task_id].to(self.device)
                    mae_error = (h * std_val - y * std_val).abs().sum().item()
                    error += mae_error
                error = error / num_graphs
                self.test_mae = error
        self.val_mae = []
        self.global_model.train()
        print (colored(f'task:{self.task_id}, current best test mae at epoch {self.current_epoch}:{self.test_mae} and {self.test_mae/conversion[self.task_id]} after normalized conversion,best validation mae:{self.best_val/conversion[self.task_id]}','red','on_yellow'))

if __name__ =='__main__':
    d = Processed_Dataset(root='../data/subgraph_count/')
    print ('num feats:',d.num_features)
    in_dim = d.num_features
    edge_dim = d.num_edge_features
    gg = d.data
    gg.y = gg.y[:,0]
    out_dim = 64
    print (d[0])
    dl = DataLoader(d,batch_size=6,shuffle=False)




