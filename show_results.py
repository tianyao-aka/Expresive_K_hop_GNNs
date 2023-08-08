from glob import glob
import pickle
import torch
import pandas as pd
from tqdm import tqdm
import argparse

def inference(validation_acc):
    validation_acc = torch.tensor(validation_acc)
    best_mean = torch.max(torch.mean(validation_acc, dim=0))
    a = best_mean.max().item()
    b = torch.argmax(torch.mean(validation_acc, dim=0)).item()
    c = validation_acc[:, b].std().item()
    best2 = torch.mean(torch.max(validation_acc, dim=1)[0]).item()
    std2 = torch.std(torch.max(validation_acc, dim=1)[0]).item()
    return a, c, b, best2, std2


def gather_dataset_stats(dataset,name):
    params = glob(dataset + '/*/*/')
    all_params_results = []
    dataset_name = dataset.split('/')[2]
    if name=='ALL': pass
    elif name!=dataset_name: return None
    for p in params:
        params_config_name = p.split('/')[-2]
        val = []
        all_folds = glob(p + '/*/**')
        for f in all_folds:
            with open(f, 'rb') as f:
                val.append(pickle.load(f))
        best1, std1, epoch, best2, std2 = inference(val)
        all_params_results.append([dataset_name, params_config_name, best1, std1, epoch, best2, std2])
    return all_params_results


def get_per_fold_stats(root_name='resultV2/separate_run/',dataset = 'ALL'):
    final_results = []
    datasets = glob(root_name + '*/')
    for dat in tqdm(datasets):
        out = gather_dataset_stats(dat,name=dataset)
        if out is None:
            continue
        final_results.extend(out)
    pd.DataFrame(final_results,columns=['dataset','params_config','best_acc_setting1','std_setting1','best_epoch','best_acc_setting2','std_setting2']).to_csv('results.csv')



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='run experiment on M2HC GNN')
    parser.add_argument('--which_dataset', type=str, default='ALL',
                        help='which dataset to use')
    args = parser.parse_args()
    dataset_name = args.which_dataset
    get_per_fold_stats(dataset=dataset_name)
    print ('done')



