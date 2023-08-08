import pickle
import sys
import os
from os import path
import shutil
import numpy as np
import argparse
from glob import glob
import torch
from calc_metrics import calc_metrics
from datetime import datetime


def get_date_str():
    x = datetime.now()
    y = str(x.month)
    d = str(x.day)
    h= str(x.hour)
    return y+d+h


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='run experiment on M2HC GNN')
    parser.add_argument('--dataset_name', type=str, default='MUTAG',
                        help='which dataset to use')
    parser.add_argument('--use_both', type=int, default=1,
                        help='use both branch or only use mhc gnn')
    parser.add_argument('--sep_conv', type=int, default=1,
                        help='weight decay')
    parser.add_argument('--feature_fusion', type=str, default='average',
                        help='weight decay')
    parser.add_argument('--combine', type=str, default='geometric',
                        help='weight decay')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')
    # parser.add_argument('--fold_index', type=int, default=0,
    #                     help='which dataset to use')

    args = parser.parse_args()
    dataset_name = 'MUTAG'
    weight_decay = 1e-6
    use_both = [True,False]
    combine = ['add','geometric']
    feature_fusion = ['average','weighted']
    sep_conv = [1,0]

    r = np.random.choice(10000)
    log_dir = f'log/{get_date_str()}/{dataset_name}/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)

    max_epoch = 350
    # layer_config = [(1, 3)]
    layer_config = [(1,5),(1,3),(2,3),(3,2),(2,4),(2,2),(3,3),(3,4)]
    layer_name = ['gcn','gin']
    log_name = dataset_name
    cnt = 0
    # layer_config = [(1, 4), (1, 3), (2, 3)]
    final_cmd = ''
    for c in combine:
        for f in feature_fusion:
            for s in sep_conv:
                for u in [False]:
                    for layer,hop in layer_config:
                        for name in layer_name:
                            out_dim = max(int(120/hop),40)
                            log_name = dataset_name
                            log_name +=f'_use_both_{use_both}_name_{name}_layer_{layer}_hop_{hop}_wd_{weight_decay}_sepconv_{sep_conv}_combine_{combine}_feature_fusion_{feature_fusion}.log'
                            if use_both:
                                cmd = f"python run_tu.py --dataset_name {dataset_name}  --out_dim {out_dim} --use_both  --base_gnn GIN --mhc_layer_num {layer} --mhc_num_hops {hop} --weight_decay {weight_decay}  --lr 5e-3  --max_epochs 350 --fold_index -1  --separate_conv {s} --jk concat --feature_fusion {f} --combine {c}  --mhc_layer_name {name} && "
                            else:
                                cmd = f"python run_tu.py --dataset_name {dataset_name}  --out_dim {out_dim} --only_mhc  --base_gnn GIN --mhc_layer_num {layer} --mhc_num_hops {hop} --weight_decay {weight_decay}  --lr 5e-3  --max_epochs 350 --fold_index -1  --separate_conv {s} --jk concat --feature_fusion {f} --combine {c}  --mhc_layer_name {name} && "
                            final_cmd += cmd
                    print ('------------------------------------------------------------------------------------')
                    print ('combine:',c,'feature fusion:',f,'sep conv:',s,'use both:',u)
                    print (final_cmd)
                    print ('\n')
                    final_cmd = ''
                    print('------------------------------------------------------------------------------------')


    # shutil.rmtree('logV2/')
    # os.mkdir('logV2/')