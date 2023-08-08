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
from model.PL_Model import PL_Model
from model.PL_Model_subgraph_count import Model
from model.base_model import GCN,GIN,GraphSAGE
import argparse
import numpy as np
from func_util import k_fold_without_validation,k_fold
import sys

sys.path.append('model/')

from dataset_processing import Processed_Dataset
from torch_geometric.loader import DataLoader
from termcolor import colored
from datetime import datetime


use_cuda =True if torch.cuda.is_available() else False
gpu_num = 1 if torch.cuda.is_available() else None


def load_dataset(dset_name):
    return Processed_Dataset(root=f'data/{dset_name}')


def get_date_str():
    x = datetime.now()
    y = str(x.month)
    d = str(x.day)
    h = str(x.hour)
    return y+d+h

def run_one_fold(dataset,out_dim,only_base_gnn,only_mhc,use_together,base_gnn_str,base_dropout,mhc_layer,mhc_dropout,base_layer,mhc_num_layers,mhc_num_hops,separate_conv,jk,feature_fusion,combine,lr,weight_decay,max_epochs,fold_index,task,use_ppgn):

    # in_dim = dataset.num_features + dataset[0].rw_feature.shape[1]   # in_dim is the concat of raw feature and rw feature
    model = Model(dataset,out_dim,only_base_gnn,only_mhc,use_together,base_gnn_str,base_dropout,mhc_layer,mhc_dropout,base_layer,mhc_num_layers,mhc_num_hops,separate_conv,jk,feature_fusion,combine,lr,weight_decay,task_id=task,use_ppgn=use_ppgn)
    a = sio.loadmat('data/subgraph_count/raw/randomgraph.mat')
    train_idx,val_idx,test_idx = a['train_idx'],a['val_idx'],a['test_idx']
    std_val = dataset.data.y[:,task].std()
    mean_val = dataset.data.y[:,task].mean()
    dataset.data.central_to_subgraph_feats *= 0.0
    dataset.data.context_samehop_feats *= 0.0
    dataset.data.y[:, task] -= mean_val   # normalize the value
    dataset.data.y[:, task] /= std_val  # normalize the value
    dataset.data.x = dataset.data.x.float()
    #trainer = pl.Trainer(max_epochs=epochs, accelerator='cpu' if not use_cuda else 'gpu', devices=1,
                         # enable_progress_bar=True)
    print(colored(f'running the {fold_index}-th fold','red','on_blue'))

    train_loader = DataLoader(dataset=dataset[train_idx],batch_size=64,shuffle=True)
    val_loader = DataLoader(dataset=dataset[val_idx],batch_size=128,shuffle=False)
    test_loader = DataLoader(dataset=dataset[test_idx], batch_size=128, shuffle=False)
    trainer = pl.Trainer(default_root_dir='saved_models/subgraph_count/',max_epochs=max_epochs,accelerator='cpu' if not use_cuda else 'gpu',devices=1,enable_progress_bar=True,logger=False,callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print ('loading')
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    best_model = model.global_model
    test_mae = []
    with torch.no_grad():
        best_model.eval()
        best_model.to('cuda')
        for batch in test_loader:
            h = best_model(batch.to('cuda'))
            h = h[:, task].view(-1,).float()
            y = batch.y[:,task].view(-1,).float()
            test_mae.append((h,y))
        pred = [i[0] for i in test_mae]
        y = [i[1] for i in test_mae]
        pred = torch.cat(pred, dim=0)
        y = torch.cat(y, dim=0)
        test_measure = F.mse_loss(pred,y)
        val = test_measure.cpu().item()
    return val



if __name__ =='__main__':
    # classification,dataset,out_dim,only_base_gnn,only_mhc,use_together,base_gnn,
    # base_dropout=0.5,mhc_dropout=0.5,base_layer=2,
    # mhc_layer=1,mhc_num_hops=3,lr=5e-3,weight_decay=1e-5,is_tudataset=False,epochs=100

    parser = argparse.ArgumentParser(description='run experiment on SEK GNN')
    parser.add_argument('--dataset_name', type=str, default='subgraph_count',
                        help='which dataset to use')
    parser.add_argument('--out_dim', type=int, default=64,
                        help='hidden dims')
    parser.add_argument('--only_base_gnn', action='store_true', default=False,
                        help='use base GNN')
    parser.add_argument('--base_gnn', type=str, default='GIN',
                        help='which base model to use')
    parser.add_argument('--only_mhc', action='store_true', default=False,
                        help='only use SEK-GNN')
    parser.add_argument('--use_both', action='store_false', default=True,
                        help='use SEK-GIN')
    parser.add_argument('--base_dropout', type=float, default=0.,
                        help='Base GNN dropout rate')
    parser.add_argument('--mhc_dropout', type=float, default=0.,
                        help='SEK GNN dropout rate')
    parser.add_argument('--base_layer_num', type=int, default=2,
                        help='Base GNN layer numbers')
    parser.add_argument('--mhc_layer_num', type=int, default=1,
                        help='SEK GNN layer numbers')
    parser.add_argument('--mhc_num_hops', type=int, default=3,
                        help='SEK-GNN number of hops per layer')
    parser.add_argument('--weight_decay', type=float, default=5e-7,
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=9e-3,
                        help='learning rate')
    parser.add_argument('--max_epochs', type=int, default=350,
                        help='max epochs')
    parser.add_argument('--result_dir', type=str, default='fill_in',
                        help='where to save results')
    parser.add_argument('--fold_index', type=int, default=-1,
                        help='fold index, not used any more')
    parser.add_argument('--separate_conv', type=int, default=1,
                        help='use separate model params for each k-hop layer or not')
    parser.add_argument('--jk', type=str, default='concat',
                        help='choice is:{concat,last}')
    parser.add_argument('--feature_fusion', type=str, default='average',
                        help='use average or weighted average for feature fusion,choice is:(average,weighted)')
    parser.add_argument('--combine', type=str, default='geometric',
                        help='geometric or add')
    parser.add_argument('--mhc_layer_name', type=str, default='gin',
                        help='base gnn encoder for SEK-GNN')
    parser.add_argument('--task', type=int, default=0,
                        help='task id')
    parser.add_argument('--use_ppgn', type=int, default=0,
                        help='not used any more')

if __name__ =='__main__':
    # x = load_dataset('MUTAG')
    # print (x[0])
    pl.seed_everything(12345)
    args = parser.parse_args()
    pyg_dataset = load_dataset(args.dataset_name)
    only_base_gnn = args.only_base_gnn
    only_mhc = args.only_mhc
    base_gnn_model = args.base_gnn
    use_both = args.use_both
    base_dropout = args.base_dropout
    mhc_dropout = args.mhc_dropout
    base_layer_num = args.base_layer_num
    mhc_layer_num = args.mhc_layer_num
    mhc_num_hops = args.mhc_num_hops
    weight_decay = args.weight_decay
    lr = args.lr
    max_epochs = args.max_epochs
    result_dir = args.result_dir
    out_dims = args.out_dim
    fold_index = args.fold_index
    sep_conv = args.separate_conv
    sep_conv = True if sep_conv==1 else False
    jk = args.jk
    feature_fusion = args.feature_fusion
    combine = args.combine
    mhc_layer_name = args.mhc_layer_name
    task = args.task
    use_ppgn = args.use_ppgn
    use_ppgn = True if use_ppgn==1 else False

    print ('dataset info:',pyg_dataset)

    params = {'dataset':pyg_dataset,'out_dim':out_dims,'only_base_gnn':only_base_gnn,'only_mhc':only_mhc,'use_together':use_both,'base_gnn_str':base_gnn_model,'base_dropout':base_dropout,'mhc_layer':mhc_layer_name,'mhc_dropout':mhc_dropout,
              'base_layer':base_layer_num,'mhc_num_layers':mhc_layer_num,'mhc_num_hops':mhc_num_hops,'separate_conv':sep_conv,'jk':jk,'feature_fusion':feature_fusion,'combine':combine,'lr':lr,'weight_decay':weight_decay,'max_epochs':max_epochs,'fold_index':fold_index,'task':task,'use_ppgn':use_ppgn}
    print ('......................start running the experiments........................')


    validation_acc = []
    records = {}
    now = get_date_str()
    params['fold_index'] = fold_index
    metrics = run_one_fold(**params)

    info=f'SEK-GIN GraphCount dataset result on {task}, test MAE loss:{metrics}'
    print (info)


    # with open(result_dir+'results.pkl','wb') as f:
    #     pickle.dump(metrics,f)
    # print (metrics)




