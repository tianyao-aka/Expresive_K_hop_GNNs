import numpy as np
import pickle
import sys
import os
from os import path
import shutil
import argparse
from glob import glob
import torch
from calc_metrics import calc_metrics
from datetime import datetime
import schedule
import nvidia_smi
from termcolor import colored

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
    parser = argparse.ArgumentParser(description='run experiment on SEK-GNN')
    parser.add_argument('--dataset_name', type=str, default='MUTAG',
                        help='which dataset to use')
    parser.add_argument('--use_both', type=int, default=1,
                        help='use SEK-GIN')
    parser.add_argument('--sep_conv', type=int, default=0,
                        help='use seperate params for each hop k in SEK-GNN')
    parser.add_argument('--feature_fusion', type=str, default='average',
                        help='average or weighted_average')
    parser.add_argument('--combine', type=str, default='geometric',
                        help='sum or geometric')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')
    parser.add_argument('--layer_num', type=int, default=2,
                        help='number of layers in SEK-GNN')
    parser.add_argument('--num_hops', type=int, default=2,
                        help='number of hops in SEK-GNN')
    parser.add_argument('--search', action='store_true', default=False,
                        help='search hyperparams for TU dataset')
    # parser.add_argument('--fold_index', type=int, default=0,
    #                     help='which dataset to use')

    args = parser.parse_args()
    dataset_name = args.dataset_name
    weight_decay = 1e-6
    use_both = [True]
    search = args.search


    r = np.random.choice(10000)
    log_dir = f'log/{get_date_str()}/{dataset_name}/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)

    max_epoch = 200
    if search:
        layer_config = [(1,3),(1,5),(2,3),(3,2),(2,4),(2,2),(3,3),(3,4)]
        layer_name = ['gcn','gin']
        combine = ['add','geometric']
        feature_fusion = ['average']
        sep_conv = [1,0]
    else:
        # only use a fixed hyper parameters
        layer_config = [(args.layer_num,args.num_hops)]
        layer_name = ['gin']
        combine = ['geometric']
        feature_fusion = ['average']
        sep_conv = [0]
    log_name = dataset_name
    cnt = 1
    # layer_config = [(1, 4), (1, 3), (2, 3)]
    final_cmd = ''
    cmd_list = []
    for c in combine:
        for f in feature_fusion:
            for s in sep_conv:
                for u in [True]:
                    for layer,hop in layer_config:
                        for name in layer_name:
                            for fo in range(10):
                                out_dim = max(int(120/hop),40)
                                log_name = dataset_name
                                log_name +=f'_use_both_{use_both}_name_{name}_layer_{layer}_hop_{hop}_wd_{weight_decay}_sepconv_{sep_conv}_combine_{combine}_feature_fusion_{feature_fusion}.log'
                                if use_both:
                                    cmd = f"python run_tu.py --fold_index {fo} --dataset_name {dataset_name}  --out_dim {out_dim} --use_both  --base_gnn GIN --mhc_layer_num {layer} --mhc_num_hops {hop} --weight_decay {weight_decay}  --lr 5e-3  --max_epochs {max_epoch}  --separate_conv {s} --jk concat --feature_fusion {f} --combine {c}  --mhc_layer_name {name} & "
                                else:
                                    cmd = f"python run_tu.py --fold_index {fo} --dataset_name {dataset_name}  --out_dim {out_dim} --only_mhc  --base_gnn GIN --mhc_layer_num {layer} --mhc_num_hops {hop} --weight_decay {weight_decay}  --lr 5e-3  --max_epochs {max_epoch}  --separate_conv {s} --jk concat --feature_fusion {f} --combine {c}  --mhc_layer_name {name} & "
                                final_cmd += cmd
                            cmd_list.append(final_cmd)
                            cnt +=1
                            final_cmd = ''
                            


    # shutil.rmtree('logV2/')
    # os.mkdir('logV2/')
    cur_cmd =0
    fail_cnt = 0
    def run_job():
        N = len(cmd_list)
        global cur_cmd,fail_cnt
        print(
            
            colored(f'Current free memory percentage:{check_memory()}. Please make sure there are 50% of free GPU memory to run the code, otherwise it will stuck. currently running {cur_cmd},there are {N - cur_cmd} remaining to run', 'red',
                    'on_yellow'))
        if cur_cmd>=N:
            print (colored('done running cmds','red','on_blue'))
            schedule.cancel_job(job)
            exit()
        if check_memory()>0.5:
            fail_cnt = 0
            code = os.system(cmd_list[cur_cmd])
            print (colored(f'code status:{code}, currently running {cur_cmd},there are {N-cur_cmd} remaining to run','red','on_yellow'))
            cur_cmd +=1
        else:
            fail_cnt +=1
            # print (colored(f'memory fail cnt inc 1:{fail_cnt}','red','on_yellow'))
            if fail_cnt>150:
                print(colored('memory full, exit program', 'red', 'on_blue'))
                schedule.cancel_job(job)
                exit()

    job=schedule.every(1).minutes.do(run_job)

    while True:
        schedule.run_pending()
    print(colored(f'there are {N - cur_cmd} remaining to run', 'red', 'on_yellow'))

