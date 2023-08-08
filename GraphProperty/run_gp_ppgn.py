import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from func_util import *
from torch_geometric.nn import GCNConv,GINConv,SAGEConv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean,scatter_sum
import scipy.io as sio
from libs.utils import SpectralDesign
from data_pna import GraphPropertyDataset
from torch_geometric.transforms import Compose
import torch_geometric.transforms as T

import argparse

class PPGN(torch.nn.Module):
    def __init__(self, nmax=30, nneuron=40,hidden2=32):
        super(PPGN, self).__init__()
        self.nmax = nmax
        self.nneuron = nneuron
        ninp = 4

        bias = False
        self.mlp1_1 = torch.nn.Conv2d(ninp, nneuron, 1, bias=bias)
        self.mlp1_2 = torch.nn.Conv2d(ninp, nneuron, 1, bias=bias)
        self.mlp1_3 = torch.nn.Conv2d(nneuron + ninp, nneuron, 1, bias=bias)

        self.mlp2_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp2_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp2_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.mlp3_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp3_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp3_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.mlp4_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp4_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp4_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.mlp5_1 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp5_2 = torch.nn.Conv2d(nneuron, nneuron, 1, bias=bias)
        self.mlp5_3 = torch.nn.Conv2d(2 * nneuron, nneuron, 1, bias=bias)

        self.h1 = torch.nn.Linear(2 * 5 * nneuron, hidden2)
        self.h2 = torch.nn.Linear(hidden2, 1)

    def forward(self, data):
        x = data.X2
        M = torch.sum(data.M, (1), True)

        x1 = F.relu(self.mlp1_1(x) * M)
        x2 = F.relu(self.mlp1_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp1_3(torch.cat([x1x2, x], 1)) * M)

        # sum or mean layer readout
        xo1 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)
        # xo1=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x1 = F.relu(self.mlp2_1(x) * M)
        x2 = F.relu(self.mlp2_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp2_3(torch.cat([x1x2, x], 1)) * M)

        # sum or mean layer readout
        xo2 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)
        # xo2=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x1 = F.relu(self.mlp3_1(x) * M)
        x2 = F.relu(self.mlp3_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp3_3(torch.cat([x1x2, x], 1)) * M)

        # sum or mean layer readout
        xo3 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)
        # xo3=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x1 = F.relu(self.mlp4_1(x) * M)
        x2 = F.relu(self.mlp4_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp4_3(torch.cat([x1x2, x], 1)) * M)

        # sum or mean layer readout
        xo4 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)
        # xo4=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x1 = F.relu(self.mlp5_1(x) * M)
        x2 = F.relu(self.mlp5_2(x) * M)
        x1x2 = torch.matmul(x1, x2) * M
        x = F.relu(self.mlp5_3(torch.cat([x1x2, x], 1)) * M)

        # sum or mean layer readout
        xo5 = torch.cat([torch.sum(x * data.M[:, 0:1, :, :], (2, 3)), torch.sum(x * data.M[:, 1:2, :, :], (2, 3))], 1)
        # xo3=torch.cat([torch.sum(x*data.M[:,0:1,:,:],(2,3))/torch.sum(data.M[:,0:1,:,:],(2,3))  ,torch.sum(x*data.M[:,1:2,:,:],(2,3))/torch.sum(data.M[:,1:2,:,:],(2,3))],1)

        x = torch.cat([xo1, xo2, xo3, xo4, xo5], 1)
        x = F.relu(self.h1(x))
        return x


class Model(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='gcn',head_num=16,random_walk_dim=5,k_hop_adj=None,random_walk_feats=None,num_classes=6,use_base_gnn = False,use_random_walk = True,use_ppgn=True,hop=3):
        super().__init__()
        self.gnn_model = GNN(in_dim, hidden1_dim,hidden2_dim,layer_name=layer_name,head_num=head_num)
        self.ppgn = PPGN(hidden2=hidden2_dim)
        if use_random_walk:
            self.subgraph_model = Subgraph_GNN(in_dim+random_walk_dim+1, hidden1_dim,hidden2_dim,k_hop_adj,random_walk_feats,use_rw= use_random_walk)
        else:
            self.subgraph_model = Subgraph_GNN(in_dim, hidden1_dim, hidden2_dim, k_hop_adj, random_walk_feats, use_rw=use_random_walk)
        # self.cls_head = nn.Linear(hidden2_dim*2,num_classes)
        self.use_rw = use_random_walk
        self.use_gnn = use_base_gnn
        self.use_ppgn = use_ppgn
        self.hop = hop

    def forward(self,x,edges,walk_feats,hop1,hop2,hop3,hop4,data):
        if self.use_gnn:
            out = self.gnn_model(x,edges)
        if self.use_ppgn:
            out = self.ppgn(data)
        if self.hop==3:
            subgraph_model_out = self.subgraph_model(x,walk_feats,hop1,hop2,hop3)
        if self.hop==4:
            subgraph_model_out = self.subgraph_model(x,walk_feats,hop1,hop2,hop3,hop4)
        subgraph_model_out = scatter_sum(subgraph_model_out, data.batch, dim=0)
        if self.use_gnn or self.use_ppgn:
            h = torch.cat((out,subgraph_model_out),dim=1)
            return h
        else:
            return subgraph_model_out



class GNN(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='gcn',head_num=16):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.name = layer_name
        if layer_name=='gcn':
            self.gcn1 = GCNConv(in_dim, hidden1_dim,add_self_loops=True)
            self.gcn2 = GCNConv(hidden1_dim, hidden2_dim,add_self_loops=True)
            self.gcn3 = GCNConv(hidden2_dim, hidden2_dim, add_self_loops=True)
            self.gcn4 = GCNConv(hidden2_dim, hidden2_dim, add_self_loops=True)
        elif layer_name=='sage':
            self.gcn1 = SAGEConv(in_dim, hidden1_dim)
            self.gcn2 = SAGEConv(hidden1_dim,hidden2_dim)
            self.gcn3 = SAGEConv(hidden2_dim, hidden2_dim)
            self.gcn4 = SAGEConv(hidden2_dim, hidden2_dim)

        elif layer_name=='gin':
            nn_callable1 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Dropout(),nn.Linear(hidden1_dim,hidden1_dim),nn.ReLU())
            nn_callable2 = nn.Sequential(nn.Linear(hidden1_dim,hidden2_dim),nn.ReLU(),nn.Dropout(),nn.Linear(hidden2_dim,hidden2_dim),nn.ReLU())
            nn_callable3 = nn.Sequential(nn.Linear(hidden2_dim,hidden2_dim),nn.ReLU(),nn.Dropout(),nn.Linear(hidden2_dim,hidden2_dim),nn.ReLU())
            nn_callable4 = nn.Sequential(nn.Linear(hidden2_dim,hidden2_dim),nn.ReLU(),nn.Dropout(),nn.Linear(hidden2_dim,hidden2_dim),nn.ReLU())
            self.gcn1 = GINConv(nn=nn_callable1)
            self.gcn2 = GINConv(nn=nn_callable2)
            self.gcn3 = GINConv(nn=nn_callable3)
            self.gcn4 = GINConv(nn=nn_callable4)
        else:
            print ('gnn module error')


    def forward(self,x,edges):
        if self.name=='gin':
            h = self.gcn1(x,edges)
            h = self.gcn2(h,edges)
            h = self.gcn3(h,edges)
            h = self.gcn4(h,edges)
        else:
            h = F.relu(self.gcn1(x,edges))
            h = F.relu(self.gcn2(h,edges))
            h = F.relu(self.gcn3(h,edges))
            h = F.relu(self.gcn4(h,edges))
        return h


class Subgraph_GNN(nn.Module):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim, k_hop_adj=[0,0,0,0],random_walk_feats=None,use_rw = True):
        super().__init__()
        self.layer0 = nn.Sequential(nn.Linear(in_dim,hidden1_dim),nn.ReLU(),nn.Linear(hidden1_dim,hidden2_dim))
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(), nn.Linear(hidden1_dim, hidden2_dim))
        self.layer2 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(), nn.Linear(hidden1_dim, hidden2_dim))
        self.layer3 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(), nn.Linear(hidden1_dim, hidden2_dim))
        self.layer4 = nn.Sequential(nn.Linear(in_dim, hidden1_dim), nn.ReLU(), nn.Linear(hidden1_dim, hidden2_dim))
        # self.hop1,self.hop2,self.hop3 = k_hop_adj[0],k_hop_adj[1],k_hop_adj[2]
        # self.rand = random_walk_feats
        self.use_rw = use_rw


    def forward(self, x,walk_feats,*hops):
        N = len(hops)
        if self.use_rw:
            X = torch.cat((x,walk_feats),dim=1)
        else:
            X = x
        hop0_out = self.layer0(X)
        hop1_out = self.layer1(hops[0].matmul(X))
        hop2_out = self.layer2(hops[1].matmul(X))
        hop3_out = self.layer3(hops[2].matmul(X))
        if N==3:
            return hop0_out + hop1_out + hop2_out + hop3_out
        if N==4:
            hop4_out = self.layer2(hops[3].matmul(X))
            return hop0_out + hop1_out + hop2_out + hop3_out + hop4_out


class PL_GCN(pl.LightningModule):
    def __init__(self,in_dim, hidden1_dim,hidden2_dim,layer_name='gcn',head_num=16,random_walk_dim=10,num_classes=6,lr=1e-2,weight_decay=2e-3,use_benchamark=False,node_classification=False,use_gnn=False,use_rw=True,task_id=0,use_ppgn=True,l1_loss=True,hop=3):
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        #         self.save_hyperparameters()
        # Create model
        self.task_id = task_id
        self.save_hyperparameters()
        self.lr = lr
        self.node_cls = node_classification
        self.weight_decay = weight_decay
        self.benchmark = use_benchamark
        self.loss = F.l1_loss if l1_loss else F.mse_loss
        if use_benchamark:
            self.model = GNN(in_dim,hidden1_dim,hidden2_dim,layer_name,head_num)
            self.cls = nn.Linear(hidden2_dim,1)
        else:
            if use_gnn or use_ppgn:
                self.cls = nn.Linear(2*hidden2_dim, 1)
            else:
                self.cls = nn.Linear(hidden2_dim, 1)
            self.args = {'in_dim': in_dim, 'hidden1_dim': hidden1_dim, 'hidden2_dim': hidden2_dim,
                         'layer_name': layer_name, 'random_walk_dim': random_walk_dim, 'num_classes': num_classes,'use_base_gnn':use_gnn,'use_random_walk':use_rw,'use_ppgn':use_ppgn,'hop':hop}
            self.model = Model(**self.args)

        self.log_prob_nn = nn.LogSoftmax(dim=-1)


    # def set_data(self, pyg_dataset):
    #     self.pyg_fulledge_dataset = pyg_dataset
    #     self.pyg_data = pyg_dataset
    #     self.calc_second_order_adj()


    def collate_graph_adj(self,edge_list, ptr):
        # print ('######################',self.device)
        edges = torch.cat([torch.tensor(i).to(self.device) + ptr[idx] for idx, i in enumerate(edge_list)], dim=1)
        N = ptr[-1]
        val = torch.tensor([1.] * edges.shape[1]).to(self.device)
        return torch.sparse_coo_tensor(edges, val, (N, N)).to(self.device)

    def forward(self,x,edges):
        # Forward function that is run when visualizing the graph
        h = self.model(x, edges)
        return h

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,20,1e-3)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        #         print (batch_idx)
        if not self.benchmark:
            hop1=self.collate_graph_adj(batch.hop1,batch.ptr)
            hop2 = self.collate_graph_adj(batch.hop2, batch.ptr)
            hop3 = self.collate_graph_adj(batch.hop3, batch.ptr)
            hop4 = self.collate_graph_adj(batch.hop4, batch.ptr)
            h = self.model(batch.x.float(), batch.edge_index,batch.rand_feature,hop1,hop2,hop3,hop4,batch)
        else:
            h = self.model(batch.x.float(), batch.edge_index)

        h=self.cls(h).view(-1,).float()
        y = batch.y[:,self.task_id].float()
        loss_val = self.loss(h,y)
        self.log("train_loss", loss_val.item(), prog_bar=True)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        #         self.log("train_acc", acc,prog_bar=True, logger=True)
        #         self.log("train_loss", loss,prog_bar=True, logger=True)
        #         self.logger.experiment.add_scalar('tree_em_loss/train',loss.item(),self.global_step)
        return loss_val  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        if not self.benchmark:
            hop1=self.collate_graph_adj(batch.hop1,batch.ptr)
            hop2 = self.collate_graph_adj(batch.hop2, batch.ptr)
            hop3 = self.collate_graph_adj(batch.hop3, batch.ptr)
            hop4 = self.collate_graph_adj(batch.hop4, batch.ptr)
            h = self.model(batch.x.float(), batch.edge_index,batch.rand_feature,hop1,hop2,hop3,hop4,batch)
        else:
            h = self.model(batch.x.float(), batch.edge_index)
        h=self.cls(h).view(-1,).float()
        y = batch.y[:,self.task_id].float()
        loss_val = self.loss(h,y)
        # By default logs it per epoch (weighted average over batches)
        self.log("val_loss", loss_val.item(), prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        if not self.benchmark:
            hop1=self.collate_graph_adj(batch.hop1,batch.ptr)
            hop2 = self.collate_graph_adj(batch.hop2, batch.ptr)
            hop3 = self.collate_graph_adj(batch.hop3, batch.ptr)
            hop4 = self.collate_graph_adj(batch.hop4, batch.ptr)
            h = self.model(batch.x.float(), batch.edge_index,batch.rand_feature,hop1,hop2,hop3,hop4,batch)
        else:
            h = self.model(batch.x.float(), batch.edge_index)
        h=self.cls(h).view(-1,)
        y = batch.y[:,self.task_id]
        loss_val = self.loss(h,y)
        # By default logs it per epoch (weighted average over batches)
        self.log("test_loss", loss_val.item(), prog_bar=True, logger=True)




if __name__=='__main__':
    # dset = Planetoid(name='CiteSeer', root='data/citeseer/')
    # model = GNN(3703, 16, 15)
    # out = model(dset.data.x, dset.data.edge_index)
    # edges = torch.tensor([[0, 1, 0, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 1, 3, 2]]).long()

    parser = argparse.ArgumentParser(description='GNN on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--task_id', type=int, default=0,
                        help='task id')
    parser.add_argument('--use_gnn', type=int, default=0,
                        help='use gnn or not')
    parser.add_argument('--use_ppgn', type=int, default=1,
                        help='use ppgn or not')
    parser.add_argument('--l1_loss', type=int, default=0,
                        help='use l2_loss or l1_loss')
    parser.add_argument('--hop', type=int, default=3,
                        help='number of hops to use in SEK-GNN')

    args = parser.parse_args()
    hop = args.hop

    results = []
    transform = SpectralDesign(nmax=30, recfield=1, dv=1, nfreq=10, adddegree=True, laplacien=False, addadj=True)
    transforms = Compose([transform])
    gp_train = GraphPropertyDataset(root='data/pna-simulation/', pre_transform=transforms,split='train')
    gp_valid = GraphPropertyDataset(root='data/pna-simulation/', pre_transform=transforms, split='val')
    gp_test = GraphPropertyDataset(root='data/pna-simulation/', pre_transform=transforms, split='test')

    train_loader = DataLoader(gp_train, batch_size=64, shuffle=True)
    valid_loader = DataLoader(gp_train, batch_size=64, shuffle=False)
    test_loader = DataLoader(gp_train, batch_size=64, shuffle=False)
    seeds = [12345,12345,12345]

    task_id = args.task_id
    l1_loss = args.l1_loss
    for i in range(1):
        seed = seeds[i]
        pl.seed_everything(seed)
        num_feats = 3
        hidden1 = 64
        hidden2 = 48
        use_gnn = True if args.use_gnn==1 else False
        use_ppgn = True if args.use_ppgn==1 else False
        use_benchmark = False
        pl_model = PL_GCN(num_feats, hidden1, hidden2, num_classes=1, use_benchamark=use_benchmark, random_walk_dim=20,node_classification=False, lr=1e-3, layer_name='gin',use_gnn=use_gnn,use_rw=True,task_id=task_id,use_ppgn=use_ppgn,l1_loss=l1_loss,hop=hop)
        trainer = pl.Trainer(default_root_dir=f'saved_models/graphproperty/', gpus=1 if torch.cuda.is_available() else 0, max_epochs=1000,devices=1,
                             callbacks=[EarlyStopping(patience=1000, monitor='val_loss', mode='min'),
                                        ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss")])
        trainer.fit(model=pl_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        model = pl_model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        res = trainer.test(model=model, dataloaders=test_loader)[0]
        res['task_id']=task_id
        res['iteration'] = i
        results.append(res)
    print (results)
    val = np.asarray([i['test_loss'] for i in results])
    mean_val = np.mean(val)
    std = np.std(val)
    print (f'stats for task {task_id}:',mean_val,std)
    msg=f'SEK-PPGN graph property metric,hop:{hop} taskId:{task_id},test_loss(log10MSE) is:{np.log10(mean_val)}'
    print (msg)



