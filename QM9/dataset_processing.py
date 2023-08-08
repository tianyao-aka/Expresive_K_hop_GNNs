import os
import torch
import sys
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}')
sys.path.append(f'{os.path.dirname(os.path.realpath(__file__))}/../')

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_geometric.utils import to_undirected,add_self_loops,remove_self_loops
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data,DataLoader
from torch_geometric.datasets import TUDataset,ZINC
import argparse
from func_util import *
from tqdm import tqdm

# def load_dataset(fpath = f'{os.path.dirname(os.path.realpath(__file__))}/data/subgraphcount_final/'):
#     dset = Processed_Dataset(fpath)
#     return dset

class Processed_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,fpath=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    def collate_graph_adj(self, edge_list, ptr):
        edges = torch.cat([torch.tensor(i) + ptr[idx] for idx, i in enumerate(edge_list)], dim=1)
        N = ptr[-1]
        return torch.sparse_coo_tensor(edges, [1.] * edges.shape[1], (N, N))

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        # data_list = self.process_data_list('data/TRIANGLE_EX/TRIANGLE_EX_test.txt')  # 记得修改这里的路径！
        if self.raw_dir[-1]=='/':
            self.raw_dir = self.raw_dir[:-1]
        self.dset_name = self.raw_dir.split('/')[-2]
        print ('dataset name:',self.dset_name)
        data_list=[]
        if self.dset_name.lower() in ['mutag','proteins','dd','bzr','nci1','enzymes','cox2'] or 'reddit' in self.dset_name.lower() or 'imdb' in self.dset_name.lower() or 'ptc' in self.dset_name.lower():
            if 'reddit' in self.dset_name.lower() or 'imdb' in self.dset_name.lower():
                pre_transform = Composer(khop_transform,degree_post_processing,pre_cache_khop_features)
            else:
                pre_transform = Composer(khop_transform, pre_cache_khop_features)

            dataset = TUDataset(root='data/tmp',name=self.dset_name)
            max_degree = degree(dataset.data.edge_index[0]).max()
            print ('max degree of the dataset is:',max_degree)
            if pre_transform:
                for d in tqdm(dataset):
                    if d.num_nodes>=1000:
                        print (d.num_nodes,'continue')
                        continue
                    data_list.append(pre_transform(d))

                # for i in range(1000):
                #     if data_list[i].num_nodes!=data_list[i].x.shape[0]:
                #         print (data_list[i])


        if 'zinc' in self.raw_dir or 'ZINC' in self.raw_dir:
            if 'train' in self.raw_dir:
                dataset = ZINC(root='data/tmp/zinc12k/',subset=True,split='train')
            elif 'val' in self.raw_dir:
                dataset = ZINC(root='data/tmp/zinc12k/', subset=True, split='val')
            else:
                dataset = ZINC(root='data/tmp/zinc12k/', subset=True, split='test')

            if pre_transform:
                print ('doing dataset pre-transform,wait for a while')
                data_list = [pre_transform(d) for d in dataset]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # def get_metric(self,idx):
    #     u = hu[idx].unsqueeze(0)
    #     out = torch.sum(u,hi,dim=-1)
    #     target_binary = [0]*91599
    #     for item_index in test_interactions[i]:
    # #             target[item_index]=rate
    #         target_binary[item_index]=1
    # #         target = torch.tensor(target)
    #     target_binary = torch.tensor(target_binary)
    #     seen = train_interations[i]
    #     out[seen]=-1.0
    #     ndcg_top20.append(retrieval_normalized_dcg(out,target_binary,k=20).item())
    # #         ndcg_top10.append(retrieval_normalized_dcg(pred,target,k=10).item())
    # #         ndcg_top30.append(retrieval_normalized_dcg(pred,target,k=30).item())
    # #         recall_at10.append(retrieval_recall(pred,target_binary,k=10).item())
    #     recall_at20.append(retrieval_recall(out,target_binary,k=20).item())
    # #         recall_at30.append(retrieval_recall(pred,target_binary,k=30).item())
    #     return recall_at20,ndcg_top20



if __name__=='__main__':
    # edges = torch.tensor([[0, 1, 0, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 1, 3, 2]]).long()
    # data_model = EdgeIndex_Processor(edges)
    # q,j = data_model.run()
    # print (q[0])
    # print (j)
    # s = Synthetic_Dataset(root='data/pyg_TRIANGLE_EX/test')
    # for d in s:
    #     if max(d.y)>1:
    #         print (d.y)
    #Processed_Dataset(root='data/TUDataset/'+'enzymes_new123')

    parser = argparse.ArgumentParser(description='data preprocessing for M2HC GNN')
    parser.add_argument('--dataset_name', type=str, default='MUTAG')
    args = parser.parse_args()
    name = args.dataset_name
    d = Processed_Dataset(root=f'data/{name}')
    print ('done data processing')
    print ('show one data sample:')
    i = d[2].nfeats
    print (d)
    print (d[2])


