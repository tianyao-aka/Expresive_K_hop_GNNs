from torch_geometric.nn import SAGEConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINEConv,GINConv, global_mean_pool, global_add_pool,GCNConv,SAGEConv
import sys
sys.path.append('../')
# from nov.dataset_processing import Processed_Dataset
from dataset_processing import Processed_Dataset
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import torch.nn as nn
import numpy as np



class SpectralDesign(object):

    def __init__(self, nmax=0, recfield=1, dv=5, nfreq=5, adddegree=False, laplacien=True, addadj=False, vmax=None):
        # receptive field. 0: adj, 1; adj+I, n: n-hop area
        self.recfield = recfield
        # b parameter
        self.dv = dv
        # number of sampled point of spectrum
        self.nfreq = nfreq
        # if degree is added to node feature
        self.adddegree = adddegree
        # use laplacian or adjacency for spectrum
        self.laplacien = laplacien
        # add adjacecny as edge feature
        self.addadj = addadj
        # use given max eigenvalue
        self.vmax = vmax

        # max node for PPGN algorithm, set 0 if you do not use PPGN
        self.nmax = nmax

    def __call__(self, data):

        n = data.x.shape[0]
        nf = data.x.shape[1]

        data.x = data.x.type(torch.float32)

        nsup = self.nfreq + 1
        if self.addadj:
            nsup += 1

        A = np.zeros((n, n), dtype=np.float32)
        SP = np.zeros((nsup, n, n), dtype=np.float32)
        A[data.edge_index[0], data.edge_index[1]] = 1

        if self.adddegree:
            data.x = torch.cat([data.x, torch.tensor(A.sum(0)).unsqueeze(-1)], 1)

        # calculate receptive field. 0: adj, 1; adj+I, n: n-hop area
        if self.recfield == 0:
            M = A
        else:
            M = (A + np.eye(n))
            for i in range(1, self.recfield):
                M = M.dot(M)

        M = (M > 0)

        d = A.sum(axis=0)
        # normalized Laplacian matrix.
        dis = 1 / np.sqrt(d)
        dis[np.isinf(dis)] = 0
        dis[np.isnan(dis)] = 0
        D = np.diag(dis)
        nL = np.eye(D.shape[0]) - (A.dot(D)).T.dot(D)
        V, U = np.linalg.eigh(nL)
        V[V < 0] = 0
        # keep maximum eigenvalue for Chebnet if it is needed
        data.lmax = V.max().astype(np.float32)

        if not self.laplacien:
            V, U = np.linalg.eigh(A)

        # design convolution supports
        vmax = self.vmax
        if vmax is None:
            vmax = V.max()

        freqcenter = np.linspace(V.min(), vmax, self.nfreq)

        # design convolution supports (aka edge features)
        for i in range(0, len(freqcenter)):
            SP[i, :, :] = M * (U.dot(np.diag(np.exp(-(self.dv * (V - freqcenter[i]) ** 2))).dot(U.T)))
            # add identity
        SP[len(freqcenter), :, :] = np.eye(n)
        # add adjacency if it is desired
        if self.addadj:
            SP[len(freqcenter) + 1, :, :] = A

        # set convolution support weigths as an edge feature
        E = np.where(M > 0)
        data.edge_index2 = torch.Tensor(np.vstack((E[0], E[1]))).type(torch.int64)
        data.edge_attr2 = torch.Tensor(SP[:, E[0], E[1]].T).type(torch.float32)

        # set tensor for Maron's PPGN
        if self.nmax > 0:
            H = torch.zeros(1, nf + 2, self.nmax, self.nmax)
            H[0, 0, data.edge_index[0], data.edge_index[1]] = 1
            H[0, 1, 0:n, 0:n] = torch.diag(torch.ones(data.x.shape[0]))
            for j in range(0, nf):
                H[0, j + 2, 0:n, 0:n] = torch.diag(data.x[:, j])
            data.X2 = H
            M = torch.zeros(1, 2, self.nmax, self.nmax)
            for i in range(0, n):
                M[0, 0, i, i] = 1
            M[0, 1, 0:n, 0:n] = 1 - M[0, 0, 0:n, 0:n]
            data.M = M
        return data