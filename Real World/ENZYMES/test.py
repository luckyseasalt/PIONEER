#imports
import torch
import torch.nn.functional as F
from torch.nn import Linear
from itertools import product
import time
from torch import tensor
from torch.optim import Adam
from torch.optim import SGD
from math import ceil
from torch.nn import Linear
from torch.distributions import categorical
from torch.distributions import Bernoulli
import torch.nn
from matplotlib import pyplot as plt
from torch_geometric.utils import convert as cnv
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
from torch.utils.data.sampler import RandomSampler
from torch.nn.functional import gumbel_softmax
from torch.distributions import relaxed_categorical
import myfuncs
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv, global_mean_pool, NNConv, GCNConv
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Batch 
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean
from torch import autograd
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
from torch_geometric.utils import dropout_adj, to_undirected, to_networkx
from torch_geometric.utils import is_undirected
from cut_utils import get_diracs
import scipy
import scipy.io
from matplotlib.lines import Line2D
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import GPUtil
from networkx.algorithms.approximation import max_clique
import pickle
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel
from torch_geometric.data import DataListLoader, DataLoader
from random import shuffle
from networkx.algorithms.approximation import max_clique
from networkx.algorithms import graph_clique_number
from networkx.algorithms import find_cliques
from torch_geometric.nn.norm import graph_size_norm
from torch_geometric.datasets import TUDataset
import visdom 
from visdom import Visdom 
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
from modules_and_utils import decode_clique_final, decode_clique_final_speed

from pre_process import connected, bridge, connected_components
from models import ELECT_Mnist
from cut_utils import get_diracs

from torch.optim import Adam
from itertools import product
import GPUtil

from torch_geometric.utils import to_dense_adj

import os,inspect
import sys

import torch
from pathlib import Path
import yaml 

import numpy as np

import argparse 
import shutil


from models import ELECT_Mnist
import random
from heuristic import heuristic
from modules_and_utils import log_msg

def kruskal(edge_list, node_num):
    
    node_list = set()
    for i in range(len(edge_list)):
        edge_list[i] = list(edge_list[i])
        node_list.add(edge_list[i][0])
        node_list.add(edge_list[i][1])
    
    old_node = sorted(list(node_list))
    
    new_node = torch.arange(node_num)
    
    mapping = dict(zip(old_node, new_node.tolist()))
    count_def = 0
    for i in edge_list:
        edge_list[count_def][0] = mapping[int(i[0])]
        edge_list[count_def][1] = mapping[int(i[1])]
        count_def += 1

    edge_num = len(edge_list)
    tree_st = []
    if node_num <= 0 or edge_num < node_num - 1:
        return tree_st
    group = [[i] for i in range(0,node_num + 1)] 
    for edge in edge_list:
        k  = len(group)
        if k == 1: 
            break
        for i in range(k): 
            if edge[0] in group[i]:
                m = i
            if edge[1] in group[i]:
                n = i
        if m != n:                
            tree_st.append(edge)  
            group[m] = group[m] + group[n] 
            group.remove(group[n])

    mapping_convert = {}
    
    for key, value in mapping.items():
        mapping_convert[value] = key
    count_def_2 = 0
    for i in tree_st:
        tree_st[count_def_2][0] = mapping_convert[int(i[0])]
        tree_st[count_def_2][1] = mapping_convert[int(i[1])]
        count_def_2 += 1

    return tree_st

def rounding(x,batch_node,discrete,adj,row_graph,col_graph,edge_attr,batch_graph,gamma,aaa,retdz,graph,zero_a,loss,test,alpha):
    for j in range(len(x[batch_node])):
        x_test = discrete[batch_node].clone()
        x_test[j] = 1 - x_test[j]
        adj_change_0 = ((1 - x_test.unsqueeze(0)- x_test.unsqueeze(1)) ** 2).squeeze(-1)
        adj_new_0 = torch.mul(adj_change_0, adj)
        laplacian_0 = torch.diag(adj_new_0.sum(dim=-1)) - adj_new_0
        eigenvalues_0 = torch.linalg.eigvals(laplacian_0) 
        eigenvalues_0, _ = torch.sort(torch.real(eigenvalues_0))
        eigenvalues_0[ eigenvalues_0 < 1e-5] = 0
        cut_edge_label_0 = torch.square(x_test[row_graph] - x_test[col_graph])* edge_attr[batch_graph] 
        pairwise_square_0 = cut_edge_label_0 # (xi-xj)^2
        solution_value_0 = (pairwise_square_0.unsqueeze(-1)).sum()/2 
        loss_0 = alpha * (gamma - solution_value_0) + gamma * torch.exp(-aaa*(eigenvalues_0[2]-eigenvalues_0[1]))
        if loss_0 < retdz["losses histogram"][0][graph]:
            discrete[batch_node] = x_test.clone()
            test = eigenvalues_0[2]
            loss = loss_0
            zero_a = len(x[batch_node])-torch.count_nonzero(eigenvalues_0)
            break
        else:
            if len(x[batch_node])-torch.count_nonzero(eigenvalues_0) <= zero_a or zero_a == 1:
                discrete[batch_node] = x_test.clone()
                test = eigenvalues_0[2]
                loss = loss_0
                zero_a = len(x[batch_node])-torch.count_nonzero(eigenvalues_0)
                continue
            elif len(x[batch_node])-torch.count_nonzero(eigenvalues_0) == 2:
                if loss >= loss_0:
                    discrete[batch_node] = x_test.clone()
                    test = eigenvalues_0[2]
                    loss = loss_0
    return loss,x_test,test,zero_a

def test(subgraph_large):

    parser = argparse.ArgumentParser(description='this is the arg parser for synthetic dataset with power grid')
    parser.add_argument('--model_path', dest = 'model_path',default = 'train_files/best_val_model_12.23_for_ENZYMES.pth')
    parser.add_argument('--datasets', dest = 'datasets',default = 'ENZYMES') 
    parser.add_argument('--gpu', dest = 'gpu',default = '3')
    parser.add_argument('--tag', default='date', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
    args = parser.parse_args()

    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date

    log_file = 'log/result_{}_gpu{}_{}.txt'.format(args.tag, args.gpu, args.datasets)

    batch_size = 32
    numlayers = 4

    test_loader = DataLoader(subgraph_large, batch_size, shuffle=False)
     
    for data in subgraph_large:
       
        num_nodes = len(data.x)
       
        edge_index_old = data.edge_index
        edge_index_new = edge_index_old.clone()
        old_node_labels_list = edge_index_old.reshape(1,-1).tolist()
        old_node_labels = sorted(list(set(old_node_labels_list[0])))
        # 重新标号节点
        new_node_labels = torch.arange(num_nodes)
        # 获取节点映射
        node_mapping = dict(zip(old_node_labels, new_node_labels.tolist()))
        for i in range(len(edge_index_old[0])):
            edge_index_new[0][i] = node_mapping[int(edge_index_old[0][i])]
            edge_index_new[1][i] = node_mapping[int(edge_index_old[1][i])]
        data.edge_index = edge_index_new

    #load the model
    model = ELECT_Mnist(subgraph_large,numlayers, 32, 1,1)
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    model_state_dict = torch.load(args.model_path, map_location = torch.device("cpu"))

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    count = 0

    not_connected = 0

    #Evaluation on test set

    mean_sum = 0
    data_num = len(subgraph_large)
    avg_value = 0
    t0 = time.time()

    for data in test_loader:

        data = data.to(device)
        data_prime = get_diracs(data.to(device), 1, device, sparse = True, effective_volume_range=0.15, receptive_field = 7)
        data_prime = data_prime.to(device)
        
        edge_index = data_prime.edge_index
        edge_attr = torch.ones(len(edge_index[0]), device = device)
        batch = data_prime.batch
        num_graphs = batch.max().item() + 1 
        row, col = edge_index 

        edge_batch_index = batch[edge_index[0]]

        retdz = model(data_prime, edge_batch_index= edge_batch_index)
        
        discrete = torch.round(retdz["output"][0])

        x = retdz["output"][0]

        loss_save = []
        gamma_save = []
        
        for graph in range(num_graphs):
            batch_graph = (edge_batch_index==graph)
            batch_node = (batch==graph)
            batch_graph_2 = torch.cat((batch_graph.view(1,-1),batch_graph.view(1,-1)),0)
            
            edge_temp = edge_index[batch_graph_2].reshape(2,-1)
            edge_temp = torch.sub(edge_temp, int(edge_temp[0][0]))
            adj = to_dense_adj(edge_temp).squeeze(0)
            adj_change_c = ((1 - x[batch_node].unsqueeze(0)- x[batch_node].unsqueeze(1)) ** 2).squeeze(-1)
            adj_new_c = torch.mul(adj_change_c, adj)
            laplacian_c = torch.diag(adj_new_c.sum(dim=-1)) - adj_new_c
            eigenvalues_c = torch.linalg.eigvals(laplacian_c)
            eigenvalues_c, _ = torch.sort(torch.real(eigenvalues_c))

            discrete[batch_node] = x[batch_node].clone()

            laplacian0 = torch.diag(adj.sum(dim=-1)) - adj
            eigenvalues0 = torch.linalg.eigh(laplacian0)
            eigenvalues0, _ = torch.sort(eigenvalues0[0])
 
            aaa = torch.log(tensor(0.0001))/(-eigenvalues0[2])
            
            if graph == 0:
                row_graph = row[batch_graph]
                col_graph = col[batch_graph]
                label = max(row_graph) + 1
            else:
                label_tensor = torch.ones(len(row[batch_graph]), device = device, dtype=torch.int64) * label
                row_graph = row[batch_graph] - label_tensor
                col_graph = col[batch_graph] - label_tensor
                label = label + max(row_graph) + 1

            alpha = len(row_graph)/(2*len(discrete[batch_node]))
            gamma = edge_attr[data.batch[data.edge_index[0]] == graph].sum()/2

            # Rounding 1
            for j in range(len(x[batch_node])):
                x_test = discrete[batch_node].clone()
                x_test[j] = 0 
                adj_change_0 = ((1 - x_test.unsqueeze(0)- x_test.unsqueeze(1)) ** 2).squeeze(-1)
                adj_new_0 = torch.mul(adj_change_0, adj)
                laplacian_0 = torch.diag(adj_new_0.sum(dim=-1)) - adj_new_0
                eigenvalues_0 = torch.linalg.eigvals(laplacian_0)
                eigenvalues_0, _ = torch.sort(torch.real(eigenvalues_0))
                eigenvalues_0[ eigenvalues_0 < 1e-5] = 0
                cut_edge_label_0 = torch.square(x_test[row_graph] - x_test[col_graph])* edge_attr[batch_graph]
                pairwise_square_0 = cut_edge_label_0 # (xi-xj)^2
                solution_value_0 = (pairwise_square_0.unsqueeze(-1)).sum()/2 
                loss_0 = alpha * (gamma - solution_value_0) + gamma * torch.exp(-aaa*(eigenvalues_0[2]-eigenvalues_0[1]))
                x_test[j] = 1
                adj_change_1 = ((1 - x_test.unsqueeze(0)- x_test.unsqueeze(1)) ** 2).squeeze(-1)
                adj_new_1 = torch.mul(adj_change_1, adj)
                laplacian_1 = torch.diag(adj_new_1.sum(dim=-1)) - adj_new_1
                eigenvalues_1 = torch.linalg.eigvals(laplacian_1) 
                eigenvalues_1, _ = torch.sort(torch.real(eigenvalues_1))
                eigenvalues_1[ eigenvalues_1 < 1e-5] = 0
                cut_edge_label_1 = torch.square(x_test[row_graph] - x_test[col_graph])* edge_attr[batch_graph]
                pairwise_square_1 = cut_edge_label_1 # (xi-xj)^2
                solution_value_1 = (pairwise_square_1.unsqueeze(-1)).sum()/2
                loss_1 = alpha * (gamma - solution_value_1) + gamma * torch.exp(-aaa*(eigenvalues_1[2]-eigenvalues_1[1]))
                if loss_1 < loss_0:
                    discrete[batch_node] = x_test.clone()
                    test = eigenvalues_1[2]
                    loss = loss_1
                    zero_a = len(x[batch_node])-torch.count_nonzero(eigenvalues_1)
                else:
                    x_test[j] = 0
                    discrete[batch_node] = x_test.clone()
                    test = eigenvalues_0[2]
                    loss = loss_0
                    zero_a = len(x[batch_node])-torch.count_nonzero(eigenvalues_0)
            
            if torch.round(loss*10)/10 >= torch.round(retdz["losses histogram"][0][graph]*10)/10: # 保留一位小数，等号成立于全标为0或全标为1的情况
                for abc in range(1000):
                    loss,x_test,test,zero_a = rounding(x,batch_node,discrete,adj,row_graph,col_graph,edge_attr,batch_graph,gamma,aaa,retdz,graph,zero_a,loss,test,alpha)
                    if loss < retdz["losses histogram"][0][graph]:
                        discrete[batch_node] = x_test.clone()
                        test = test
                        break

            
            loss_save.append(float(loss))
            gamma_save.append(float(alpha*gamma))

            # Rounding 2
            if test < 1e-4:
                discrete[batch_node] = x[batch_node].clone()
                for j in range(len(x[batch_node])):
                    x_test = discrete[batch_node].clone()
                    x_test[j] = 0
                    adj_change_0 = ((1 - x_test.unsqueeze(0)- x_test.unsqueeze(1)) ** 2).squeeze(-1)
                    adj_new_0 = torch.mul(adj_change_0, adj)
                    laplacian_0 = torch.diag(adj_new_0.sum(dim=-1)) - adj_new_0
                    eigenvalues_0 = torch.linalg.eigvals(laplacian_0)
                    eigenvalues_0, _ = torch.sort(torch.real(eigenvalues_0))
                    test_0 = eigenvalues_0[2] - eigenvalues_c[2]
                    x_test[j] = 1
                    adj_change_1 = ((1 - x_test.unsqueeze(0)- x_test.unsqueeze(1)) ** 2).squeeze(-1)
                    adj_new_1 = torch.mul(adj_change_1, adj)
                    laplacian_1 = torch.diag(adj_new_1.sum(dim=-1)) - adj_new_1
                    eigenvalues_1 = torch.linalg.eigvals(laplacian_1)
                    eigenvalues_1, _ = torch.sort(torch.real(eigenvalues_1))
                    test_1 = eigenvalues_1[2] - eigenvalues_c[2]
                    if test_1 > test_0:
                        discrete[batch_node] = x_test.clone()
                        test = eigenvalues_1[2]
                    else:
                        x_test[j] = 0
                        discrete[batch_node] = x_test.clone()
                        test = eigenvalues_0[2]

        
        cut_edge_label = torch.square(discrete[row] - discrete[col])
        pairwise_square = cut_edge_label # (xi-xj)^2
        
        for graph in range(num_graphs):
            connected_label = 1
            batch_graph = (edge_batch_index==graph)
            batch_node = (batch==graph)
            solution_value = (pairwise_square[batch_graph].unsqueeze(-1)).sum()/2 
            batch_graph_2 = torch.cat((batch_graph.view(1,-1),batch_graph.view(1,-1)),0)
            index_bi = edge_index[batch_graph_2].reshape(2,-1)
            cut_edge_graph_bi = cut_edge_label[batch_graph]
            
            connections = []
            num_count = 0
            node_indices_graph = torch.unique(index_bi) 
            node_indices_graph = set(node_indices_graph.tolist())
            for j in cut_edge_graph_bi:
                if j == 0:
                    if int(index_bi[0][num_count]) < int(index_bi[1][num_count]):
                        connections.append((int(index_bi[0][num_count]),int(index_bi[1][num_count])))
                    else:
                        connections.append((int(index_bi[1][num_count]),int(index_bi[0][num_count])))
                num_count += 1
            connections = list(set(connections))         
            G = nx.Graph()
            G.add_edges_from(connections)
            
            node_indices = set(G.nodes())
            diff = node_indices_graph.difference(node_indices)
            if len(diff) != 0: 
                G.add_nodes_from(list(diff))
            components = connected_components(G)
            if len(components) != 2:
                not_connected += 1
            else:
                mean_sum = mean_sum + solution_value

            # heuristic：
            connections_initial = []
            num_count_2 = 0
            weight = {}
            graph_edge_attr = edge_attr[batch_graph]
            for j in range(len(cut_edge_graph_bi)):
                if int(index_bi[0][num_count_2]) < int(index_bi[1][num_count_2]):
                    temp_edge = (int(index_bi[0][num_count_2]),int(index_bi[1][num_count_2]))
                    connections_initial.append(temp_edge)
                    weight[temp_edge]=float(graph_edge_attr[j])
                else:
                    temp_edge = (int(index_bi[1][num_count_2]),int(index_bi[0][num_count_2]))
                    connections_initial.append(temp_edge)
                num_count_2 += 1
            connections_initial = list(set(connections_initial))
            initial_graph = nx.Graph()
            initial_graph.add_edges_from(connections_initial)
            
            for edge in initial_graph.edges():
                if weight.get((edge[0],edge[1])) != None: 
                    initial_graph[edge[0]][edge[1]]['weight'] = weight.get((edge[0],edge[1]))
                else:
                    initial_graph[edge[0]][edge[1]]['weight'] = weight.get((edge[1],edge[0]))

            sub_trees = []
            for component in components:
                subgraph = initial_graph.subgraph(component)
                subgraph = list(subgraph.edges())
                random.shuffle(subgraph)
                sub_tree = kruskal(subgraph, len(component))
                sub_trees = sub_trees + sub_tree
            num_count_3 = 0
            for j in sub_trees:
                if j[0] < j[1]:
                    sub_trees[num_count_3] =(j[0],j[1])
                else:
                    sub_trees[num_count_3] =(j[1],j[0])
                num_count_3 += 1
            
            diff_edge = list(set(connections_initial) - set(sub_trees))
            random.shuffle(diff_edge)
            edge_tree = sub_trees + diff_edge
            tree = kruskal(edge_tree, len(x[batch_node]))
            initial_tree = nx.Graph()
            initial_tree.add_edges_from(tree)
            max_cut, best_value = heuristic(initial_graph, initial_tree)
            avg_value = avg_value + best_value
            
        count += 1

    t1 = (time.time() - t0)/float(data_num)
    avg_value = avg_value/float(data_num)  
    mean_sum = mean_sum/float(data_num-not_connected) 
    print("solution not connected:" + str(not_connected))
    print("solution value mean:" + str(mean_sum)) 
    print("elect solution value mean:" + str(avg_value))
    print("elect time:"+str(t1)) 

    # output
    msg = f'solution not connected: {not_connected},\n unsupervised value mean:{mean_sum}, \n elect value mean:{avg_value},\n elect time: {t1}\n'
    log_msg(msg, log_file)

    return not_connected,mean_sum,avg_value, t1