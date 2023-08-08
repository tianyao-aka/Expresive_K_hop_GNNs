import json
import math
import os
import sys
import scipy.io as sio
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append('data/')
sys.path.append('model/')
import torch
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import path
import shutil
import time
import json
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch_geometric.loader import DataLoader
# # pytorch lightning
from torch_geometric.datasets import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math
from model.PL_Model_QM9 import Model
from model.base_model import GCN,GIN,GraphSAGE
import argparse
import numpy as np
import sys
from dataset_processing import Processed_Dataset
from torch_geometric.loader import DataLoader
from termcolor import colored
from datetime import datetime
from tqdm import tqdm

use_cuda =True if torch.cuda.is_available() else False
gpu_num = 1 if torch.cuda.is_available() else None
device = 'cuda' if use_cuda else 'cpu'


def get_date_str():
    x = datetime.now()
    y = str(x.month)
    d = str(x.day)
    h = str(x.hour)
    return y+d+h
# out_dim,only_base_gnn,only_mhc,use_together,base_gnn_str,base_dropout=0.5,mhc_layer='gcn',mhc_dropout=0.5,base_layer=2,mhc_num_layers=1,mhc_num_hops=3,separate_conv = True,jk = 'concat',feature_fusion='weighted',combine='geometric',lr=1e-3,weight_decay=1e-6,virtual_node=False,task_id=0,pooling_method='sum'
def run_one_fold(out_dim,only_base_gnn,only_mhc,use_together,base_gnn_str,base_dropout,mhc_layer,mhc_dropout,base_layer,mhc_num_layers,mhc_num_hops,separate_conv,jk,feature_fusion,combine,lr,weight_decay,max_epochs,task,virtual_node,pooling_method,**kargs):
    dataset = QM9('data/QM9_processed/')
    dataset = dataset.shuffle()

    tenpercent = int(len(dataset) * 0.1)
    mean = dataset.data.y[tenpercent:].mean(dim=0)
    std = dataset.data.y[tenpercent:].std(dim=0)
    dataset.data.y = (dataset.data.y - mean) / std
    test_dataset = dataset[:tenpercent]
    val_dataset = dataset[tenpercent:2 * tenpercent]
    train_dataset = dataset[2 * tenpercent:]
    test_loader = DataLoader(test_dataset, batch_size=128)
    val_loader = DataLoader(val_dataset, batch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    model = Model(dataset,out_dim=out_dim,only_base_gnn=only_base_gnn,only_mhc=only_mhc,use_together=use_together,base_gnn_str=base_gnn_str,base_dropout=base_dropout,mhc_layer=mhc_layer,mhc_dropout=mhc_dropout,base_layer=base_layer,mhc_num_layers=mhc_num_layers,mhc_num_hops=mhc_num_hops,separate_conv=separate_conv,jk=jk,feature_fusion=feature_fusion,combine=combine,lr=lr,weight_decay=weight_decay,task_id=task,virtual_node=virtual_node,pooling_method=pooling_method,test_loader=test_loader,std=std)

    trainer = pl.Trainer(default_root_dir='saved_models/qm9/',max_epochs=max_epochs,accelerator='cpu' if not use_cuda else 'gpu',devices=1,enable_progress_bar=True,logger=False)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return model.test_mae



if __name__ =='__main__':
    # out_dim,only_base_gnn,only_mhc,use_together,base_gnn_str,base_dropout,mhc_layer,mhc_dropout,base_layer,mhc_num_layers,
    # mhc_num_hops,separate_conv,jk,feature_fusion,combine,lr,weight_decay,max_epochs,task,virtual_node,pooling_method

    parser = argparse.ArgumentParser(description='run experiment on M2HC GNN')
    parser.add_argument('--out_dim', type=int, default=64,
                        help='which dataset to use')
    parser.add_argument('--only_base_gnn', action='store_true', default=False,
                        help='use base GNN')
    parser.add_argument('--base_gnn_str', type=str, default='GIN',
                        help='which base model to use')
    parser.add_argument('--only_mhc', action='store_true', default=False,
                        help='use MHC GNN')
    parser.add_argument('--use_together', action='store_false', default=True,
                        help='use MHC GNN+Base GNN')
    parser.add_argument('--base_dropout', type=float, default=0.,
                        help='Base GNN dropout rate')
    parser.add_argument('--mhc_dropout', type=float, default=0.,
                        help='MHC GNN dropout rate')
    parser.add_argument('--base_layer', type=int, default=5,
                        help='Base GNN layer numbers')
    parser.add_argument('--mhc_num_layers', type=int, default=1,
                        help='MHC GNN layer numbers')
    parser.add_argument('--mhc_num_hops', type=int, default=3,
                        help='MHC GNN number hops per layer')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='max epochs')
    parser.add_argument('--fold_index', type=int, default=1,
                        help='fold index')
    parser.add_argument('--separate_conv', type=int, default=1,
                        help='use separate model params for each k-hop layer or not')
    parser.add_argument('--jk', type=str, default='attention',
                        help='concat or last')
    parser.add_argument('--feature_fusion', type=str, default='average',
                        help='use add or weighted sum for feature fusion')
    parser.add_argument('--combine', type=str, default='geometric',
                        help='geometric or add')
    parser.add_argument('--mhc_layer', type=str, default='gin',
                        help='geometric or add')
    parser.add_argument('--task', type=int, default=0,
                        help='geometric or add')
    parser.add_argument('--virtual_node', action='store_false', default=True,
                        help='geometric or add')
    parser.add_argument('--pooling_method', type=str, default='attention',
                        help='geometric or add')

if __name__ =='__main__':
    # x = load_dataset('MUTAG')
    # print (x[0])
    from datetime import datetime
    pl.seed_everything(12345)
    args = parser.parse_args()
    args = vars(args)
    metrics = run_one_fold(**args)
    task = args['task']




