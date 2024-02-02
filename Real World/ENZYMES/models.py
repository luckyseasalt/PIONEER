##############################
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import tensor
from torch.optim import Adam
from torch.optim import SGD
from math import ceil
from torch.nn import Linear
from torch.distributions import categorical
from torch.distributions import Bernoulli
import torch.nn
from torch_geometric.utils import convert as cnv
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.data import Batch 
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean
from torch import autograd
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops, dropout_adj
from modules_and_utils import get_diracs, get_mask, propagate
from modules_and_utils import derandomize_cut, GATAConv, get_diracs
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import to_scipy_sparse_matrix, add_self_loops, get_laplacian,to_dense_adj

from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
import math
import numpy as np
###########

device = torch.device("cuda:"+str(3) if torch.cuda.is_available() else "cpu") 

def total_var(x, edge_index, batch, undirected = True):
    row, col = edge_index
    if undirected:
        tv = (torch.abs(x[row]-x[col])) * 0.5
    else:
        tv = (torch.abs(x[row]-x[col]))

    tv = scatter_add(tv, batch[row], dim=0)
    return  tv

class ELECT_Mnist(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden1, hidden2, deltas):
        super(ELECT_Mnist, self).__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.momentum = 0.1
        self.convs = torch.nn.ModuleList()
        self.deltas = deltas
        self.numlayers = num_layers
        self.heads = 8
        self.concat = True
        
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.bns.append(BN(self.heads*self.hidden1, momentum=self.momentum))
        self.convs = torch.nn.ModuleList()        
        for i in range(num_layers - 1):
                self.convs.append(GINConv(Sequential(
            Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
            ReLU(),
            Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
            ReLU(),
            BN(self.heads*self.hidden1, momentum=self.momentum),
        ),train_eps=True))
        self.bn1 = BN(self.heads*self.hidden1)       
        self.conv1 = GINConv(Sequential(Linear(self.hidden2,  self.heads*self.hidden1),
            ReLU(),
            Linear( self.heads*self.hidden1,  self.heads*self.hidden1),
            ReLU(),
            BN(self.heads*self.hidden1, momentum=self.momentum),
        ),train_eps=True)

        if self.concat:
            self.lin1 = Linear(self.heads*self.hidden1, self.hidden1)
        else:
            self.lin1 = Linear(self.hidden1, self.hidden1)
        self.lin2 = Linear(self.hidden1, 1)
        self.gnorm = GraphSizeNorm()  # graph size normalization


    def reset_parameters(self):
        self.conv1.reset_parameters()
        
        for conv in self.convs:
            conv.reset_parameters() 
        for bn in self.bns:
            bn.reset_parameters()
        self.bn1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data, edge_dropout = None, penalty_coefficient_f = 1, penalty_coefficient_g = 1, edge_batch_index  = None):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        num_graphs = batch.max().item() + 1 
        row, col = edge_index     
        total_num_edges = edge_index.shape[1]
        N_size = x.shape[0] 

        
        if edge_dropout is not None:
            edge_index = dropout_adj(edge_index, edge_attr = (torch.ones(edge_index.shape[1], device=device)).long(), p = edge_dropout, force_undirected=True)[0]
            edge_index = add_remaining_self_loops(edge_index, num_nodes = batch.shape[0])[0]

        # no_loop_index,_ = remove_self_loops(edge_index) 
        # no_loop_row, no_loop_col = no_loop_index 

        x = x.unsqueeze(-1)
        # mask = get_mask(x,edge_index,1).to(x.dtype)
        x = F.leaky_relu(self.conv1(x, edge_index))# +x  # num_node_in_batch*512
        # x = x*mask
        x = self.gnorm(x)
        x = self.bn1(x)
        
        for conv, bn in zip(self.convs, self.bns):
            if(x.dim()>1):
                x =  x + F.leaky_relu(conv(x, edge_index))
                # mask = get_mask(mask,edge_index,1).to(x.dtype)
                # x = x*mask
                x = self.gnorm(x)
                x = bn(x)

        x = F.leaky_relu(self.lin1(x)) # num_node_in_batch*64
        # x = x*mask

        x = F.leaky_relu(self.lin2(x)) # num_node_in_batch*1
        # x = x*mask


        #calculate min and max
        batch_max = scatter_max(x, batch, 0, dim_size= N_size)[0]
        batch_max = torch.index_select(batch_max, 0, batch)        
        batch_min = scatter_min(x, batch, 0, dim_size= N_size)[0]
        batch_min = torch.index_select(batch_min, 0, batch)


        #min-max normalize
        x = (x-batch_min)/(batch_max+1e-4-batch_min)

        if torch.isnan(x).any() == True: 
            print("aa")
            
        probs = x

        pairwise_prodsums = torch.zeros(num_graphs, device = device)
        gamma = edge_batch_index.bincount(minlength=data.num_graphs)/2 

        pairwise_square = torch.square(probs[row] - probs[col]) # (xi-xj)^2 
       

        lamma = torch.zeros(num_graphs, device = device) 
        aaa = torch.zeros(num_graphs, device = device) 
        alpha = torch.ones(num_graphs, device = device) 

        for graph in range(num_graphs): 
            batch_graph = (edge_batch_index==graph)
            batch_node = (batch==graph)
            pairwise_prodsums[graph] = (pairwise_square[batch_graph].unsqueeze(-1)).sum()/2 
            batch_graph_2 = torch.cat((batch_graph.view(1,-1),batch_graph.view(1,-1)),0)
            
            edge_temp = edge_index[batch_graph_2].reshape(2,-1) 
            edge_temp = torch.sub(edge_temp, int(edge_temp[0][0])) 
            adj = to_dense_adj(edge_temp).squeeze(0)

            laplacian0 = torch.diag(adj.sum(dim=-1)) - adj
            eigenvalues0 = torch.linalg.eigh(laplacian0) 
            eigenvalues0, _ = torch.sort(eigenvalues0[0]) 

            aaa[graph] = torch.log(tensor(0.0001))/(-eigenvalues0[2]) 

            adj_change = ((1 - probs[batch_node].unsqueeze(0)- probs[batch_node].unsqueeze(1)) ** 2).squeeze(-1)
            adj_new = torch.mul(adj_change, adj)  
            laplacian = torch.diag(adj_new.sum(dim=-1)) - adj_new
            eigenvalues = torch.linalg.eigh(laplacian) 
            eigenvalues, _ = torch.sort(eigenvalues[0])
            lamma[graph] = eigenvalues[2] - eigenvalues[1]
            alpha[graph] = len(pairwise_square[batch_graph])/(2*len(probs[batch_node]))

            
        # f
        f_relax = gamma - pairwise_prodsums 
  
        # g
        g_relax = torch.exp(-aaa*lamma)
        
        # calculate loss
        loss = alpha * f_relax + gamma * g_relax 

        retdict = {}
        
        retdict["output"] = [probs.squeeze(-1),"hist"]   #output
        retdict["losses histogram"] = [loss.squeeze(-1),"hist"]
        retdict["loss"] = [loss.mean().squeeze(),"sequence"] #final loss
        retdict["f"] = [f_relax.squeeze(-1),"cost"]
        retdict["g"] = [g_relax.squeeze(-1),"constrain"]
        retdict["gamma"] = [(gamma).mean().squeeze(),"gamma"]

        return retdict
    
    def __repr__(self):
        return self.__class__.__name__
