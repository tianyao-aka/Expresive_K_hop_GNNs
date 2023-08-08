import pickle
import sys
import os
from os import path
import shutil
import argparse
from glob import glob
import torch
from datetime import datetime

def calc_metrics(fpath,is_tu,is_cls,dataset_name,params):
    fnames = glob(f'{fpath}/{dataset_name}/{params}/**/*.pkl',recursive=True)
    records = []
    for q in fnames:
        with open(q,'rb') as f:
            val = pickle.load(f)
        records.append(val)
    if is_tu:
        validation_acc = torch.tensor(records)
        # print (validation_acc)
        # print ('record shape:',validation_acc.shape)
        best_mean = torch.max(torch.mean(validation_acc,dim=0))
        a = best_mean.max().item()
        b = torch.argmax(torch.mean(validation_acc,dim=0)).item()
        c = validation_acc[:,b].std().item()
        best2 = torch.mean(torch.max(validation_acc, dim=1)[0]).item()
        std2 = torch.std(torch.max(validation_acc, dim=1)[0]).item()
        print ('best validation mean acc:',a,' at epoch:',b,'std:',c)
        info = f"best valid acc at epoch_{b} is {a}, std is:{c};  best mean valid acc is:{best2}, std is:{std2}"
    else:
        if is_cls:  #
            rec = torch.tensor(records).view(-1,)
            mean_acc = rec.mean()
            std_acc  = rec.std()
            info = f"test mean acc is:{mean_acc},std is:{std_acc}"
        else:
            rec = torch.tensor(records).view(-1,)
            mean_mae = rec.mean()
            std_mae  = rec.std()
            info = f"test mean acc is:{mean_mae},std is:{std_mae}"
    if path.exists(f'result/{dataset_name}_{params}.txt'):
        os.remove(f'result/{dataset_name}_{params}.txt')
    with open(f'result/{dataset_name}_{params}.txt','w') as f:
        f.writelines(info)
    print (f'final result for {name} {params} is: {info}')


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='calculate final metrics')
    parser.add_argument('--fpath', type=str, default='result',
                        help='path to save result')
    parser.add_argument('--is_tu', type=int, default=1,
                        help='')
    parser.add_argument('--is_cls', type=int, default=1,
                        help='which dataset to use')

    parser.add_argument('--dataset_name', type=str, default=None,
                        help='which dataset to use')
    parser.add_argument('--params', type=str, default=None,
                        help='model params')
    # parser.add_argument('--fold_index', type=int, default=0,
    #                     help='which dataset to use')

    args = parser.parse_args()
    fpath = args.fpath
    is_tu = args.is_tu
    is_cls = args.is_cls
    name = args.dataset_name
    params = args.params
    calc_metrics(fpath,is_tu,is_cls,name,params)
    print ('finish time on calc_metrics:',datetime.now())



