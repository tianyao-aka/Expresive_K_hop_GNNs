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
import schedule
import nvidia_smi
from termcolor import colored
from send_email import send_email

def check_memory():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    nvidia_smi.nvmlShutdown()
    return info.free/info.total



def get_date_str():
    x = datetime.now()
    y = str(x.month)
    d = str(x.day)
    h= str(x.hour)
    return y+d+h

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='run experiment on M2HC GNN')
    parser.add_argument('--dataset_name', type=str, default='subgraph_count_ppgn',
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
    parser.add_argument('--task', type=int, default=0,
                        help='task')
    # parser.add_argument('--fold_index', type=int, default=0,
    #                     help='which dataset to use')

    args = parser.parse_args()
    dataset_name = args.dataset_name
    weight_decay = 1e-6
    use_both = [True]
    # combine = ['add','geometric']
    # feature_fusion = ['average','weighted']
    combine = ['geometric']
    feature_fusion = ['average']
    sep_conv = [1]

    max_epoch = 350
    # layer_config = [(1, 3)]
    layer_config = [(6,3),(7,5),(6,5)]
    layer_name = ['gin']
    log_name = dataset_name
    cnt = 1
    tasks = [0,1]
    # layer_config = [(1, 4), (1, 3), (2, 3)]
    cmd_list = []
    for c in combine:
        for f in feature_fusion:
            for s in sep_conv:
                for u in [True]:
                    for layer,hop in layer_config:
                        for name in layer_name:
                            for task in tasks:
                                for fo in range(7,10):
                                    out_dim = 64
                                    log_name = dataset_name
                                    log_name +=f'_use_both_{use_both}_name_{name}_layer_{layer}_hop_{hop}_wd_{weight_decay}_sepconv_{sep_conv}_combine_{combine}_feature_fusion_{feature_fusion}.log'
                                    if use_both:
                                        cmd = f" python run_subgraph_count.py --fold_index {fo} --dataset_name {dataset_name}  --out_dim {out_dim} --use_both  --base_gnn GIN --mhc_layer_num {layer} --mhc_num_hops {hop} --weight_decay {weight_decay}  --lr 9e-3  --max_epochs {max_epoch}  --separate_conv {s} --jk concat --feature_fusion {f} --combine {c}  --mhc_layer_name {name} --task {task} --use_ppgn 0 & "
                                    else:
                                        cmd = f" python run_subgraph_count.py --fold_index {fo} --dataset_name {dataset_name}  --out_dim {out_dim} --only_mhc  --base_gnn GIN --mhc_layer_num {layer} --mhc_num_hops {hop} --weight_decay {weight_decay}  --lr 9e-3  --max_epochs {max_epoch}  --separate_conv {s} --jk concat --feature_fusion {f} --combine {c}  --mhc_layer_name {name} --task {task} --use_ppgn 0 & "
                                    cmd_list.append(cmd)
                                    print ('------------------------------------------------------------------------------------')
                                    print ('combine:',c,'feature fusion:',f,'sep conv:',s,'use both:',u,'layer_config:',layer,hop,name)
                                    cnt +=1
                                    print ('\n')
                                    print('------------------------------------------------------------------------------------')


    # shutil.rmtree('logV2/')
    # os.mkdir('logV2/')
    cur_cmd =0
    fail_cnt = 0
    print (len(cmd_list))
    def run_job():
        N = len(cmd_list)
        global cur_cmd,fail_cnt
        print(
            colored(f'currently running {cur_cmd},there are {N - cur_cmd} remaining to run', 'red',
                    'on_yellow'))
        if check_memory()>0.15:
            fail_cnt = 0
            code = os.system(cmd_list[cur_cmd])
            print (colored(f'code status:{code}, currently running {cur_cmd},there are {N-cur_cmd} remaining to run','red','on_yellow'))
            cur_cmd +=1
            send_email(info='update subgraph counting dataset',msg=f'code status:{code}, currently running {cur_cmd},there are {N-cur_cmd} remaining to run')
        if cur_cmd>=N:
            print (colored('done running cmds','red','on_blue'))
            schedule.cancel_job(job)
            exit()

        else:
            fail_cnt +=1
            print (colored(f'memory fail cnt inc 1:{fail_cnt}','red','on_yellow'))
            if fail_cnt>2000:
                print(colored('memory full, exit program', 'red', 'on_blue'))
                schedule.cancel_job(job)
                exit()

    job=schedule.every(1).minutes.do(run_job)

    while True:
        schedule.run_pending()
    print(colored(f'there are {N - cur_cmd} remaining to run', 'red', 'on_yellow'))


