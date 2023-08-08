import json
import math
import os
import sys
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
from model.PL_Model_TUDataset_V2 import PL_TU_Model
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

def run_one_fold(dataset,out_dim,only_base_gnn,only_mhc,use_together,base_gnn_str,base_dropout,mhc_layer,mhc_dropout,base_layer,mhc_num_layers,mhc_num_hops,separate_conv,jk,feature_fusion,combine,lr,weight_decay,max_epochs,fold_index):

    # in_dim = dataset.num_features + dataset[0].rw_feature.shape[1]   # in_dim is the concat of raw feature and rw feature
    model = PL_TU_Model(dataset,out_dim,only_base_gnn,only_mhc,use_together,base_gnn_str,base_dropout,mhc_layer,mhc_dropout,base_layer,mhc_num_layers,mhc_num_hops,separate_conv,jk,feature_fusion,combine,lr,weight_decay)
    train_idx,test_idx = k_fold_without_validation(dataset,10)
    validation_acc = []
    #trainer = pl.Trainer(max_epochs=epochs, accelerator='cpu' if not use_cuda else 'gpu', devices=1,
                         # enable_progress_bar=True)
    print(colored(f'running the {fold_index}-th fold','red','on_blue'))
    train_loader = DataLoader(dataset=dataset[train_idx[fold_index]],batch_size=64,shuffle=True)
    val_loader = DataLoader(dataset=dataset[test_idx[fold_index]],batch_size=64,shuffle=False)
    trainer = pl.Trainer(max_epochs=max_epochs,accelerator='cpu' if not use_cuda else 'gpu',devices=1,enable_progress_bar=True,logger=False)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    validation_acc.append(model.record_acc)
    print (colored(f"best valid acc for fold:{fold_index}:{torch.tensor(model.record_acc).view(-1,).max().item()}",'red','on_blue'))
    return model.record_acc


if __name__ =='__main__':
    # classification,dataset,out_dim,only_base_gnn,only_mhc,use_together,base_gnn,
    # base_dropout=0.5,mhc_dropout=0.5,base_layer=2,
    # mhc_layer=1,mhc_num_hops=3,lr=5e-3,weight_decay=1e-5,is_tudataset=False,epochs=100

    parser = argparse.ArgumentParser(description='run experiment on M2HC GNN')
    parser.add_argument('--dataset_name', type=str, default='MUTAG',
                        help='which dataset to use')
    parser.add_argument('--out_dim', type=int, default=64,
                        help='which dataset to use')
    parser.add_argument('--only_base_gnn', action='store_true', default=False,
                        help='use base GNN')
    parser.add_argument('--base_gnn', type=str, default='GIN',
                        help='which base model to use')
    parser.add_argument('--only_mhc', action='store_true', default=False,
                        help='use MHC GNN')
    parser.add_argument('--use_both', action='store_true', default=False,
                        help='use MHC GNN+Base GNN')
    parser.add_argument('--base_dropout', type=float, default=0.5,
                        help='Base GNN dropout rate')
    parser.add_argument('--mhc_dropout', type=float, default=0.5,
                        help='MHC GNN dropout rate')
    parser.add_argument('--base_layer_num', type=int, default=2,
                        help='Base GNN layer numbers')
    parser.add_argument('--mhc_layer_num', type=int, default=1,
                        help='MHC GNN layer numbers')
    parser.add_argument('--mhc_num_hops', type=int, default=3,
                        help='MHC GNN number hops per layer')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=8e-3,
                        help='learning rate')
    parser.add_argument('--max_epochs', type=int, default=350,
                        help='max epochs')
    parser.add_argument('--result_dir', type=str, default='fill_in',
                        help='where to save results')
    parser.add_argument('--fold_index', type=int, default=-1,
                        help='fold index')
    parser.add_argument('--separate_conv', type=int, default=1,
                        help='use separate model params for each k-hop layer or not')
    parser.add_argument('--jk', type=str, default='concat',
                        help='concat or last')
    parser.add_argument('--feature_fusion', type=str, default='weighted or average',
                        help='use add or weighted sum for feature fusion')
    parser.add_argument('--combine', type=str, default='geometric',
                        help='geometric or add')
    parser.add_argument('--mhc_layer_name', type=str, default='gin',
                        help='geometric or add')

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

    print ('dataset info:',pyg_dataset)

    params = {'dataset':pyg_dataset,'out_dim':out_dims,'only_base_gnn':only_base_gnn,'only_mhc':only_mhc,'use_together':use_both,'base_gnn_str':base_gnn_model,'base_dropout':base_dropout,'mhc_layer':mhc_layer_name,'mhc_dropout':mhc_dropout,
              'base_layer':base_layer_num,'mhc_num_layers':mhc_layer_num,'mhc_num_hops':mhc_num_hops,'separate_conv':sep_conv,'jk':jk,'feature_fusion':feature_fusion,'combine':combine,'lr':lr,'weight_decay':weight_decay,'max_epochs':max_epochs,'fold_index':fold_index}
    print ('......................start running the experiments........................')

    validation_acc = []
    records = {}
    if fold_index==-1:
        print(colored(f'run all TEN folds', 'red'))
        for fold in range(10):
            params['fold_index']=fold
            metrics = run_one_fold(**params)
            validation_acc.append(metrics)
        validation_acc = torch.tensor(validation_acc)
        best_mean = torch.max(torch.mean(validation_acc,dim=0))
        a = best_mean.max().item()
        b = torch.argmax(torch.mean(validation_acc,dim=0)).item()
        c = validation_acc[:,b].std().item()
        best2 = torch.mean(torch.max(validation_acc, dim=1)[0]).item()
        std2 = torch.std(torch.max(validation_acc, dim=1)[0]).item()
        records['best_acc'] = a
        records['std'] = c
        records['epoch'] = b
        records['best_acc_v2'] = best2
        records['std_v2'] = std2
        info = f'For dataset {args.dataset_name}, the best validation mean acc is:{a} at epoch {b}, std is:{c}. ver2: best mean validation accuracy is:{best2},std is: {std2}'
        print (colored(info,'red'))
        now = get_date_str()
        file_params = f'date_{now}_{args.dataset_name}_base_gnn_{base_gnn_model}_mhc_layer_name_{mhc_layer_name}_layer_{mhc_layer_num}_hop_{mhc_num_hops}_useboth_{use_both}_sepconv_{sep_conv}_jk_{jk}_feature_fusion_{feature_fusion}_combine_{combine}_wd_{weight_decay}_lr_{lr}.txt'
        with open(f'resultV2/{file_params}','w') as f:
            f.writelines(info)
        print('parameters:', params)
    else:
        print (colored(f'only run {fold_index}-th fold','red'))
        now = get_date_str()
        params['fold_index'] = fold_index
        metrics = run_one_fold(**params)
        saved_path = f'resultV2/separate_run/{args.dataset_name}/date_{now}/params_{args.dataset_name}_base_gnn_{base_gnn_model}_mhc_layer_name_{mhc_layer_name}_layer_{mhc_layer_num}_hop_{mhc_num_hops}_useboth_{use_both}_sepconv_{sep_conv}_jk_{jk}_feature_fusion_{feature_fusion}_combine_{combine}_wd_{weight_decay}_lr_{lr}/fold_{fold_index}/'
        if not path.exists(saved_path):
            os.makedirs(saved_path,exist_ok=True)
        with open(saved_path+'results.pkl','wb') as f:
            pickle.dump(metrics,f)



    # with open(result_dir+'results.pkl','wb') as f:
    #     pickle.dump(metrics,f)
    # print (metrics)



