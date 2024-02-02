##############################
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch import tensor
from math import ceil
from torch.nn import Linear
import torch.nn
from torch_geometric.utils import convert as cnv
from torch_geometric.utils import sparse as sp
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import GINConv, GATConv
from torch.nn import Parameter
from torch.nn import Sequential as Linear, ReLU
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_scatter import scatter_min, scatter_max, scatter_add
from torch_geometric.utils import add_remaining_self_loops, dropout_adj
from torch_geometric.utils import to_dense_adj

from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm

###########

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


    def forward(self, data, edge_dropout = None, penalty_coefficient = 0.25, edge_batch_index  = None):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = data.edge_attr
        edge_attr = edge_attr.to(torch.float)
        num_graphs = batch.max().item() + 1 
        row, col = edge_index     
        total_num_edges = edge_index.shape[1]
        N_size = x.shape[0] 

        
        if edge_dropout is not None:
            edge_index = dropout_adj(edge_index, edge_attr = (torch.ones(edge_index.shape[1], device=device)).long(), p = edge_dropout, force_undirected=True)[0]
            edge_index = add_remaining_self_loops(edge_index, num_nodes = batch.shape[0])[0]

        x = x.unsqueeze(-1)
        # mask = get_mask(x,edge_index,1).to(x.dtype)
        x = F.leaky_relu(self.conv1(x, edge_index))# +x
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

        x = F.leaky_relu(self.lin1(x)) 
        # x = x*mask

        x = F.leaky_relu(self.lin2(x)) 
        # x = x*mask


        #calculate min and max
        batch_max = scatter_max(x, batch, 0, dim_size= N_size)[0]
        batch_max = torch.index_select(batch_max, 0, batch)        
        batch_min = scatter_min(x, batch, 0, dim_size= N_size)[0]
        batch_min = torch.index_select(batch_min, 0, batch)

        #min-max normalize
        x = (x-batch_min)/(batch_max+1e-6-batch_min)

        probs = x 

        pairwise_prodsums = torch.zeros(num_graphs, device = device)

        gamma = torch.zeros(num_graphs, device = device)
        pairwise_square = torch.square(probs[row] - probs[col]) * edge_attr.reshape(-1,1) # wij*(xi-xj)^2

        adj = to_dense_adj(data.edge_index, batch = data.batch) 
        adj = adj.to(torch.float)
        adj_new = adj.clone()
        lamma3 = torch.zeros(num_graphs, device = device) 
        aaa = torch.ones(num_graphs, device = device) 
        alpha = torch.ones(num_graphs, device = device) 
        for graph in range(num_graphs): 
            batch_graph = (edge_batch_index==graph)
            batch_node = (batch==graph)
            
            pairwise_prodsums[graph] = (pairwise_square[batch_graph].unsqueeze(-1)).sum()/2
            gamma[graph] = edge_attr[data.batch[data.edge_index[0]] == graph].sum()/2 
            
            laplacian0 = torch.diag(adj[graph].sum(dim=-1)) - adj[graph]
            eigenvalues0 = torch.linalg.eigh(laplacian0)
            eigenvalues0, _ = torch.sort(eigenvalues0[0])

            aaa[graph] = torch.log(tensor(0.0001))/(-eigenvalues0[2])

            adj_change = ((1 - probs[batch_node].unsqueeze(0)- probs[batch_node].unsqueeze(1)) ** 2).squeeze(-1)
            adj_new[graph] = torch.mul(adj_change, adj[graph]) 
            laplacian = torch.diag(adj_new[graph].sum(dim=-1)) - adj_new[graph]
            laplacian = laplacian.to(torch.float)
            eigenvalues = torch.linalg.eigh(laplacian) 
            eigenvalues, _ = torch.sort(eigenvalues[0]) 
            lamma3[graph] = eigenvalues[2] - eigenvalues[1]
            alpha[graph] = len(pairwise_square[batch_graph])/(2*36.)
            
        # f
        f_relax = gamma - pairwise_prodsums 
  
        # g
        g_relax = torch.exp(-aaa*lamma3)
        
        # calculate loss
        loss = alpha * f_relax + gamma * g_relax 


        retdict = {}
        
        retdict["output"] = [probs.squeeze(-1),"hist"]   #output
        retdict["losses histogram"] = [loss.squeeze(-1),"hist"]
        retdict["loss"] = [loss.mean().squeeze(),"sequence"] #final loss
        retdict["f"] = [f_relax.squeeze(-1),"cost"]
        retdict["g"] = [g_relax.squeeze(-1),"constrain"]

        return retdict
    
    def __repr__(self):
        return self.__class__.__name__
