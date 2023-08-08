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
from model.PL_Model_TUDataset import PL_TU_Model
from model.base_model import GCN,GIN,GraphSAGE
import argparse
import numpy as np
from func_util import k_fold_without_validation,k_fold
import sys
sys.path.append('model/')

from nov.dataset_processing import Processed_Dataset
from torch_geometric.loader import DataLoader
from termcolor import colored

use_cuda =True if torch.cuda.is_available() else False
gpu_num = 1 if torch.cuda.is_available() else None


def load_dataset(dset_name):
    return Processed_Dataset(root=f'data/{dset_name}')

def run_one_fold(classification,dataset,out_dim,only_base_gnn,only_mhc,use_together,base_gnn,base_dropout=0.5,mhc_dropout=0.5,base_layer=2,mhc_layer=1,mhc_num_hops=3,weight_decay=1e-5,is_tudataset=False,epochs=100,lr=5e-3,fold_index=0):
    # pl.seed_everything(12345)
    num_classes = dataset.num_classes

    in_dim = dataset.num_features + dataset[0].rw_feature.shape[1]   # in_dim is the concat of raw feature and rw feature
    print ('in dim of the dataset:',in_dim)
    # if is_tudataset:
    #     model = PL_TU_Model(num_classes,classification,dataset,in_dim, out_dim,only_base_gnn,only_mhc,use_together,base_gnn,base_dropout,mhc_dropout,base_layer,mhc_layer,mhc_num_hops,lr,weight_decay)
    # else:
    #     model = PL_Model(num_classes,classification,dataset,in_dim, out_dim,only_base_gnn,only_mhc,use_together,base_gnn,base_dropout,mhc_dropout,base_layer,mhc_layer,mhc_num_hops,lr,weight_decay)
    if is_tudataset:
        train_idx,test_idx = k_fold_without_validation(dataset,10)
        validation_acc = []
        records = {}
        model = PL_TU_Model(num_classes, classification, dataset, in_dim, out_dim, only_base_gnn, only_mhc,
                            use_together, base_gnn, base_dropout, mhc_dropout, base_layer, mhc_layer, mhc_num_hops,
                            lr, weight_decay)

        #trainer = pl.Trainer(max_epochs=epochs, accelerator='cpu' if not use_cuda else 'gpu', devices=1,
                             # enable_progress_bar=True)
        print(colored(f'running the {fold_index}-th fold','red','on_blue'))
        # model.global_model.reset_parameters()
        train_loader = DataLoader(dataset=dataset[train_idx[fold_index]],batch_size=64,shuffle=True,num_workers=8)
        val_loader = DataLoader(dataset=dataset[test_idx[fold_index]],batch_size=64,shuffle=False,num_workers=8)
        trainer = pl.Trainer(max_epochs=epochs,accelerator='cpu' if not use_cuda else 'gpu',devices=1,enable_progress_bar=True,logger=False,log_every_n_steps=1e10)
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        validation_acc.append(model.record_acc)
        print (colored(f"best valid acc for fold:{fold_index}:{torch.tensor(model.record_acc).view(-1,).max().item()}",'red','on_blue'))
        model.global_model.reset_parameters()
        return model.record_acc
        # best_mean = torch.max(torch.mean(validation_acc,dim=0))
        # a = best_mean.max().item()
        # b = torch.argmax(torch.mean(validation_acc,dim=0)).item()
        # c = validation_acc[:,b].std().item()
        # best2 = torch.mean(torch.max(validation_acc, dim=1)[0]).item()
        # std2 = torch.std(torch.max(validation_acc, dim=1)[0]).item()
        # print ('best validation mean acc:',a,' at epoch:',b,'std:',c)
        # records['best_acc'] = a
        # records['std'] = c
        # records['epoch'] = b
        # records['best_acc_v2'] = best2
        # records['std_v2'] = std2
        # print ('results summary')
        # print (records)
    else:
        train_idx,val_idx,test_idx = k_fold(dataset,10)
        model = PL_Model(num_classes, classification, dataset, in_dim, out_dim, only_base_gnn, only_mhc,
                         use_together, base_gnn, base_dropout, mhc_dropout, base_layer, mhc_layer, mhc_num_hops, lr,
                         weight_decay)
        print(colored(f'running the {fold_index}-th fold','red','on_blue'))
        train_loader = DataLoader(dataset=dataset[train_idx[fold_index]],batch_size=64,shuffle=True)
        val_loader = DataLoader(dataset=dataset[val_idx[fold_index]],batch_size=64,shuffle=False)
        test_loader = DataLoader(dataset=dataset[test_idx[fold_index]], batch_size=64, shuffle=False)
        if classification:
            trainer = pl.Trainer(devices=gpu_num,accelerator='gpu' if use_cuda else 'cpu',max_epochs=epochs,callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")])
        else:
            trainer = pl.Trainer(devices=gpu_num,
                                 accelerator='gpu' if use_cuda else 'cpu', max_epochs=epochs,
                                 callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])

        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        res = trainer.test(model=model,dataloaders=test_loader)[0]
        model.global_model.reset_parameters()
        return res


if __name__ =='__main__':
    # classification,dataset,out_dim,only_base_gnn,only_mhc,use_together,base_gnn,
    # base_dropout=0.5,mhc_dropout=0.5,base_layer=2,
    # mhc_layer=1,mhc_num_hops=3,lr=5e-3,weight_decay=1e-5,is_tudataset=False,epochs=100

    parser = argparse.ArgumentParser(description='run experiment on M2HC GNN')
    parser.add_argument('--classification', action='store_true', default=False,
                        help='is classification or regression problem')
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
    parser.add_argument('--is_tudataset', action='store_true', default=False,
                        help='use tudataset or other')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='max epochs')
    parser.add_argument('--fold_index', type=int, default=0,
                        help='The fold using for exp')
    parser.add_argument('--result_dir', type=str, default='fill_in',
                        help='where to save results')

if __name__ =='__main__':
    # x = load_dataset('MUTAG')
    # print (x[0])
    args = parser.parse_args()
    is_cls = args.classification
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
    is_tudataset = args.is_tudataset
    max_epochs = args.max_epochs
    fold_index = args.fold_index
    result_dir = args.result_dir
    out_dims = args.out_dim
    print ('dataset info:',pyg_dataset)
    print ('show one data sample:',pyg_dataset[0])
    params = {'classification':is_cls,'dataset':pyg_dataset,'out_dim':out_dims,'only_base_gnn':only_base_gnn,'only_mhc':only_mhc,
             'use_together':use_both,'base_gnn':base_gnn_model,'base_dropout':base_dropout,'mhc_dropout':mhc_dropout,'base_layer':base_layer_num,'mhc_layer':mhc_layer_num,'mhc_num_hops':mhc_num_hops,
            'lr':lr,'weight_decay':weight_decay,'is_tudataset':is_tudataset,'epochs':max_epochs,'fold_index':fold_index}
    print ('......................start running the experiments........................')
    print ('parameters used:',params)
    for _ in range(2):
        print ('abcde')
        metrics = run_one_fold(**params)
        print('parameters:', params)

    # with open(result_dir+'results.pkl','wb') as f:
    #     pickle.dump(metrics,f)
    # print (metrics)



    # pyg_dataset = load_dataset('MUTAG')
    # train_idx, test_idx = k_fold_without_validation(pyg_dataset, 10)
    # pl_model = PL_TU_Model(2,True,pyg_dataset,10000, 64,True,False,False,'GCN',0.5,0.5,2,1,3,5e-3,1e-5)
    # trainer = pl.Trainer(default_root_dir=f'saved_model/{435}',
    #                      accelerator='cpu', max_epochs=150)
    # tr = DataLoader(pyg_dataset[train_idx[0]],batch_size=60)
    # trainer.fit(model=pl_model, train_dataloaders=tr, val_dataloaders=tr)


