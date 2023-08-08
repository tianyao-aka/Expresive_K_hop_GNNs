import os, torch, numpy as np
import pickle
import graph_algorithms
from graph_generation import GraphType, generate_graph
from inspect import signature

from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import dense_to_sparse

from func_util import *
from torch_geometric import transforms as T
from torch_geometric.utils import degree
from tqdm import tqdm


def collate_graph_adj(edge_list, ptr):
    edges = torch.cat([torch.tensor(i) + ptr[idx] for idx, i in enumerate(edge_list)], dim=1)
    N = ptr[-1]
    return torch.sparse_coo_tensor(edges,[1.]*edges.shape[1], (N, N))


class EdgeIndex_Processor():
    def __init__(self, edge_index,num_nodes):
        super().__init__()
        self.random_walk = None
        self.num_nodes = num_nodes
        adj,N = self.to_sparse_tensor(edge_index)
        adj_with_selfloop = self.to_sparse_tensor_with_selfloop(edge_index)
        self.N = N
        self.adj = adj.float()
        self.adj_with_loop = adj_with_selfloop.float()
        self.k_hop_neibrs = [adj.float()]
        self.calc_random_walk_matrix()

    def to_sparse_tensor(self, edge_index):
        edge_index = remove_self_loops(edge_index)[0]
        r = len(edge_index[0])
        N = self.num_nodes
        t = torch.sparse_coo_tensor(edge_index, [1] * r, (N, N))
        return t, N

    def to_sparse_tensor_with_selfloop(self, edge_index):
        edge_index = add_self_loops(edge_index)[0]
        r = len(edge_index[0])
        N = self.num_nodes
        t = torch.sparse_coo_tensor(edge_index, [1] * r, (N, N))
        return t

    def calc_random_walk_matrix(self):
        t = self.adj_with_loop.to_dense().sum(dim=1)
        t[t==0]=1.
        t = 1./t
        n = self.num_nodes
        ind = torch.tensor([[i,i] for i in range(n)]).T
        diag = torch.sparse_coo_tensor(ind,t,(n,n))
        random_walk = torch.sparse.mm(diag,self.adj)
        self.random_walk = random_walk

    def calc_random_walk_feature(self,order=10):
        t = self.random_walk
        tot_walk_feats = []
        walk_feats = []
        for i in range(self.N):
            walk_feats.append(t[i,i])
        tot_walk_feats.append(walk_feats)
        for i in range(order):
            walk_feats = []
            t = torch.sparse.mm(t,self.random_walk)
            for i in range(self.N):
                walk_feats.append(t[i, i])
            tot_walk_feats.append(walk_feats)
        tot_walk_feats = torch.tensor(tot_walk_feats).T
        return tot_walk_feats


    def calc_adj_power(self,adj, power):
        t = adj
        for _ in range(power - 1):
            t = torch.sparse.mm(t, adj)
        # set value to one
        indices = t.coalesce().indices()
        v = t.coalesce().values()
        v = torch.tensor([1 if i > 1 else i for i in v])
        diag_mask = indices[0] != indices[1]
        indices = indices[:, diag_mask]
        v = v[diag_mask]
        t = torch.sparse_coo_tensor(indices, v, (self.N, self.N))
        return t

    def postprocess_k_hop_neibrs(self,sparse_adj):
        diag = torch.diag(1. / sparse_adj.to_dense().sum(dim=1))
        diag = diag.to_sparse()
        out = torch.sparse.mm(diag, sparse_adj)
        return out


    def calc_k_hop_neibrs(self,k_hop=2):
        adj_hop_k = self.calc_adj_power(self.adj, k_hop)
        one_hop = self.k_hop_neibrs[0]
        prev_hop = self.k_hop_neibrs[1:k_hop]
        for p in prev_hop:
            one_hop += p
        final_res = adj_hop_k - one_hop

        indices = final_res.coalesce().indices()
        v = final_res.coalesce().values()
        v = [0 if i <= 0 else 1 for i in v]
        masking = []
        v_len = len(v)
        for i in range(v_len):
            if v[i] > 0:
                masking.append(i)
        v = torch.tensor(v)
        masking = torch.tensor(masking).long()
        indices = indices[:, masking]
        v = v[masking]
        final_res = torch.sparse_coo_tensor(indices, v, (self.N, self.N))
        return final_res


    def run(self,k_hop=[2,3,4],random_walk_order=20):
        walk_feature = self.calc_random_walk_feature(order=random_walk_order)
        for k in k_hop:
            t = self.calc_k_hop_neibrs(k)
            self.k_hop_neibrs.append(t.float())
        # normed_k_hop_adj = [self.postprocess_k_hop_neibrs(i.float()) for i in self.k_hop_neibrs]   # 是否使用D^-1*A
        return self.k_hop_neibrs,walk_feature


class GraphPropertyDataset(InMemoryDataset):
    # parameters for generating the dataset
    seed=12345
    graph_type='RANDOM'
    extrapolation=False
    nodes_labels=["eccentricity", "graph_laplacian_features", "sssp"]
    graph_labels = ["is_connected", "diameter", "spectral_radius"]

    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["generated_data.pkl"]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        # generate dataset
        print("Generating dataset...")
        genereate_dataset(root=self.raw_dir, seed=self.seed, graph_type=self.graph_type,
                          extrapolation=self.extrapolation, 
                          nodes_labels=self.nodes_labels, 
                          graph_labels=self.graph_labels)

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            (adj, features, node_labels, graph_labels) = pickle.load(f)
        # normalize labels
        max_node_labels = torch.cat([nls.max(0)[0].max(0)[0].unsqueeze(0) for nls in node_labels['train']]).max(0)[0]
        max_graph_labels = torch.cat([gls.max(0)[0].unsqueeze(0) for gls in graph_labels['train']]).max(0)[0]
        for dset in node_labels.keys():
            node_labels[dset] = [nls / max_node_labels for nls in node_labels[dset]]
            graph_labels[dset] = [gls / max_graph_labels for gls in graph_labels[dset]]

        graphs = to_torch_geom(adj, features, node_labels, graph_labels)
        for key, data_list in graphs.items():
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in tqdm(data_list)]
            final_data_list = []
            for t in tqdm(data_list):
                q, j = EdgeIndex_Processor(t.edge_index, t.num_nodes).run()
                hop1, hop2, hop3, hop4 = q[0], q[1], q[2], q[3]
                t.rand_feature = j
                x2 = torch.concat((t.x[:, [0]], j), dim=1)
                hop1_feature = hop1.matmul(x2)
                hop2_feature = hop2.matmul(x2)
                hop3_feature = hop3.matmul(x2)
                hop4_feature = hop4.matmul(x2)

                hop1 = hop1.coalesce().indices().tolist()
                hop2 = hop2.coalesce().indices().tolist()
                hop3 = hop3.coalesce().indices().tolist()
                hop4 = hop4.coalesce().indices().tolist()
                t.hop1 = hop1
                t.hop2 = hop2
                t.hop3 = hop3
                t.hop4 = hop4
                t.hop1_feature = hop1_feature
                t.hop2_feature = hop2_feature
                t.hop3_feature = hop3_feature
                t.hop4_feature = hop4_feature
                final_data_list.append(t)

            data, slices = self.collate(final_data_list)
            torch.save((data, slices), os.path.join(self.processed_dir, f'{key}.pt'))



def to_torch_geom(adj, features, node_labels, graph_labels):
    graphs = {}
    for key in adj.keys():      # train, val, test
        graphs[key] = []
        for i in range(len(adj[key])):          # Graph of a given size
            batch_i = []
            for j in range(adj[key][i].shape[0]):       # Number of graphs
                graph_adj = adj[key][i][j]
                graph = Data(x=features[key][i][j],
                             edge_index=dense_to_sparse(graph_adj)[0],
                             y=graph_labels[key][i][j].unsqueeze(0),
                             pos=node_labels[key][i][j])
                batch_i.append(graph)

            graphs[key].extend(batch_i)
    return graphs

def genereate_dataset(root='data', seed=12345, graph_type='RANDOM', extrapolation=False,
                      nodes_labels=["eccentricity", "graph_laplacian_features", "sssp"],
                      graph_labels = ["is_connected", "diameter", "spectral_radius"]):
 
    if not os.path.exists(root):
        os.makedirs(root)

    if 'sssp' in nodes_labels:
        sssp = True
        nodes_labels.remove('sssp')
    else:
        sssp = False

    nodes_labels_algs = list(map(lambda s: getattr(graph_algorithms, s), nodes_labels))
    graph_labels_algs = list(map(lambda s: getattr(graph_algorithms, s), graph_labels))

    def get_nodes_labels(A, F, initial=None):
        labels = [] if initial is None else [initial]

        for f in nodes_labels_algs:
            params = signature(f).parameters
            labels.append(f(A, F) if 'F' in params else f(A))
        return np.swapaxes(np.stack(labels), 0, 1)

    def get_graph_labels(A, F):
        labels = []
        for f in graph_labels_algs:
            params = signature(f).parameters
            labels.append(f(A, F) if 'F' in params else f(A))
        return np.asarray(labels).flatten()  

    GenerateGraphPropertyDataset(n_graphs={'train': [512] * 10, 'val': [128] * 5, 'default': [256] * 5},
                                N={**{'train': range(15, 25), 'val': range(15, 25)}, **(
                                    {'test-(20,25)': range(20, 25), 'test-(25,30)': range(25, 30),
                                        'test-(30,35)': range(30, 35), 'test-(35,40)': range(35, 40),
                                        'test-(40,45)': range(40, 45), 'test-(45,50)': range(45, 50),
                                        'test-(60,65)': range(60, 65), 'test-(75,80)': range(75, 80),
                                        'test-(95,100)': range(95, 100)} if extrapolation else
                                    {'test': range(15, 25)})},
                                seed=seed, graph_type=getattr(GraphType, graph_type),
                                get_nodes_labels=get_nodes_labels, get_graph_labels=get_graph_labels,
                                sssp=True, filename=f"{root}/generated_data.pkl")

class GenerateGraphPropertyDataset:
    def __init__(self, n_graphs, N, seed, graph_type, get_nodes_labels, get_graph_labels, print_every=20, sssp=True, filename="./data/multitask_dataset.pkl"):
        self.adj = {}
        self.features = {}
        self.nodes_labels = {}
        self.graph_labels = {}

        def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd=""):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix), end=printEnd)

        def to_categorical(x, N):
            v = np.zeros(N)
            v[x] = 1
            return v

        for dset in N.keys():
            if dset not in n_graphs:
                n_graphs[dset] = n_graphs['default']

            total_n_graphs = sum(n_graphs[dset])

            set_adj = [[] for _ in n_graphs[dset]]
            set_features = [[] for _ in n_graphs[dset]]
            set_nodes_labels = [[] for _ in n_graphs[dset]]
            set_graph_labels = [[] for _ in n_graphs[dset]]
            generated = 0

            progress_bar(0, total_n_graphs, prefix='Generating {:20}\t\t'.format(dset),
                         suffix='({} of {})'.format(0, total_n_graphs))

            for batch, batch_size in enumerate(n_graphs[dset]):
                for i in range(batch_size):
                    # generate a random graph of type graph_type and size N
                    seed += 1
                    adj, features, type = generate_graph(N[dset][batch], graph_type, seed=seed)

                    while np.min(np.max(adj, 0)) == 0.0:
                        # remove graph with singleton nodes
                        seed += 1
                        adj, features, _ = generate_graph(N[dset][batch], type, seed=seed)

                    generated += 1
                    if generated % print_every == 0:
                        progress_bar(generated, total_n_graphs, prefix='Generating {:20}\t\t'.format(dset),
                                     suffix='({} of {})'.format(generated, total_n_graphs))

                    # make sure there are no self connection
                    assert np.all(
                        np.multiply(adj, np.eye(N[dset][batch])) == np.zeros((N[dset][batch], N[dset][batch])))

                    if sssp:
                        # define the source node
                        source_node = np.random.randint(0, N[dset][batch])

                    # compute the labels with graph_algorithms; if sssp add the sssp
                    node_labels = get_nodes_labels(adj, features,
                                                   graph_algorithms.all_pairs_shortest_paths(adj, 0)[source_node]
                                                   if sssp else None)
                    graph_labels = get_graph_labels(adj, features)
                    if sssp:
                        # add the 1-hot feature determining the starting node
                        features = np.stack([to_categorical(source_node, N[dset][batch]), features], axis=1)

                    set_adj[batch].append(adj)
                    set_features[batch].append(features)
                    set_nodes_labels[batch].append(node_labels)
                    set_graph_labels[batch].append(graph_labels)

            self.adj[dset] = [torch.from_numpy(np.asarray(adjs)).float() for adjs in set_adj]
            self.features[dset] = [torch.from_numpy(np.asarray(fs)).float() for fs in set_features]
            self.nodes_labels[dset] = [torch.from_numpy(np.asarray(nls)).float() for nls in set_nodes_labels]
            self.graph_labels[dset] = [torch.from_numpy(np.asarray(gls)).float() for gls in set_graph_labels]
            progress_bar(total_n_graphs, total_n_graphs, prefix='Generating {:20}\t\t'.format(dset),
                         suffix='({} of {})'.format(total_n_graphs, total_n_graphs), printEnd='\n')

        self.save_as_pickle(filename)

    def save_as_pickle(self, filename):
        """" Saves the data into a pickle file at filename """
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump((self.adj, self.features, self.nodes_labels, self.graph_labels), f)


if __name__ == '__main__':
    dataset = GraphPropertyDataset(root='data/pna-simulation', split='train')
    print (dataset[1])

