import numpy as np
import torch
from torch import tensor

from torch_geometric.data import DataLoader, Data

import argparse

from pathlib import Path
import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
import sys
sys.path.append("../")
import yaml

from build_dataset.build_data import Synthetic_Mnist_Dataset
from train_atheta.models import ELECT_Mnist
from algorithms import Solution, connected_components, UnionFind

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from cut_utils import get_diracs
import time
import random


from torch_geometric.utils import to_dense_adj
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


def main():

    parser = argparse.ArgumentParser(description='this is the arg parser for synthetic dataset with mnist')
    parser.add_argument('--model_path', dest = 'model_path',default = '../train_atheta/train_files/best_val_model.pth')
    parser.add_argument('--testset_path', dest = 'testset_path',default = '../build_dataset/testset_cover/')
    parser.add_argument('--gpu', dest = 'gpu',default = '5')
    parser.add_argument('--tag', default='date', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
    args = parser.parse_args()

    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date

    log_file = 'log/result_{}_gpu{}_synthetic dataset.txt'.format(args.tag, args.gpu)

    batch_size = 32
    numlayers = 4

    cfg = Path("../build_dataset/configs/config.yaml") 
    cfg_dict = yaml.safe_load(cfg.open('r'))
    dataset = Synthetic_Mnist_Dataset(cfg_dict['data'])
    data_splits = dataset.get_idx_split()
    test_dataset = dataset[data_splits['test']]
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    #load the model
    model = ELECT_Mnist(dataset,numlayers, 32, 1,1) 
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    model_state_dict = torch.load(args.model_path, map_location = torch.device("cpu"))

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    count = 1

    not_connected = 0

    mean_sum = 0
    mean_sum_h = 0
    t1 = 0

    #Evaluation on test set

    t0 = time.time()

    for data in test_loader:
        data = data.to(device)
        data_prime = get_diracs(data.to(device), 1, device, sparse = True, effective_volume_range=0.15, receptive_field = 7)
        data_prime = data_prime.to(device)
        
        edge_index = data_prime.edge_index
        batch = data_prime.batch
        num_graphs = batch.max().item() + 1
        edge_attr = data_prime.edge_attr
        row, col = edge_index 

        edge_batch_index = data_prime.batch[data_prime.edge_index[0]]

        retdz = model(data_prime, edge_batch_index = edge_batch_index)
        
        discrete = torch.round(retdz["output"][0]) 

        xx = retdz["output"][0]
        adj = to_dense_adj(edge_index, batch = batch)
        adj_new_c = adj.clone()


        for graph in range(num_graphs):
            batch_graph = (edge_batch_index==graph)
            batch_node = (batch==graph)

            adj_change_c = ((1 - xx[batch_node].unsqueeze(0)- xx[batch_node].unsqueeze(1)) ** 2).squeeze(-1)
            adj_new_c[graph] = torch.mul(adj_change_c, adj[graph])  
            laplacian_c = torch.diag(adj_new_c[graph].sum(dim=-1)) - adj_new_c[graph]
            eigenvalues_c = torch.linalg.eigvals(laplacian_c) 
            eigenvalues_c, _ = torch.sort(torch.real(eigenvalues_c))

            laplacian0 = torch.diag(adj[graph].sum(dim=-1)) - adj[graph]
            eigenvalues0 = torch.linalg.eigh(laplacian0) 
            eigenvalues0, _ = torch.sort(eigenvalues0[0]) 

            aaa = torch.log(tensor(0.0001))/(-eigenvalues0[2]) 

            discrete[batch_node] = xx[batch_node].clone()

            if graph == 0:
                row_graph = row[batch_graph]
                col_graph = col[batch_graph]
                label = max(row_graph) + 1
            else:
                label_tensor = torch.ones(len(row[batch_graph]), device = device, dtype=torch.int64) * label
                row_graph = row[batch_graph] - label_tensor
                col_graph = col[batch_graph] - label_tensor
                label = label + max(row_graph) + 1
            
            # Rounding 1
            alpha = len(row_graph)/(2*len(discrete[batch_node]))

            gamma = edge_attr[data.batch[data.edge_index[0]] == graph].sum()/2
            for j in range(len(xx[batch_node])):
                x_test = discrete[batch_node].clone()
                x_test[j] = 0 
                adj_change_0 = ((1 - x_test.unsqueeze(0)- x_test.unsqueeze(1)) ** 2).squeeze(-1)
                adj_new_0 = torch.mul(adj_change_0, adj[graph])
                laplacian_0 = torch.diag(adj_new_0.sum(dim=-1)) - adj_new_0
                eigenvalues_0 = torch.linalg.eigvals(laplacian_0) 
                eigenvalues_0, _ = torch.sort(torch.real(eigenvalues_0))  
                cut_edge_label_0 = torch.square(x_test[row_graph] - x_test[col_graph])* edge_attr[batch_graph] 
                pairwise_square_0 = cut_edge_label_0 # (xi-xj)^2
                solution_value_0 = (pairwise_square_0.unsqueeze(-1)).sum()/2 
                loss_0 = alpha * (gamma - solution_value_0) + gamma * torch.exp(-aaa*(eigenvalues_0[2]))
                x_test[j] = 1
                adj_change_1 = ((1 - x_test.unsqueeze(0)- x_test.unsqueeze(1)) ** 2).squeeze(-1)
                adj_new_1 = torch.mul(adj_change_1, adj[graph])
                laplacian_1 = torch.diag(adj_new_1.sum(dim=-1)) - adj_new_1
                eigenvalues_1 = torch.linalg.eigvals(laplacian_1)
                eigenvalues_1, _ = torch.sort(torch.real(eigenvalues_1)) 
                cut_edge_label_1 = torch.square(x_test[row_graph] - x_test[col_graph])* edge_attr[batch_graph] 
                pairwise_square_1 = cut_edge_label_1 # (xi-xj)^2
                solution_value_1 = (pairwise_square_1.unsqueeze(-1)).sum()/2 
                loss_1 = alpha * (gamma - solution_value_1) + gamma * torch.exp(-aaa*(eigenvalues_1[2]))
                if loss_1 < loss_0:
                    discrete[batch_node] = x_test.clone()
                    test = eigenvalues_1[2]
                else:
                    x_test[j] = 0
                    discrete[batch_node] = x_test.clone()
                    test = eigenvalues_0[2]
         
        cut_edge_label = torch.square(discrete[row] - discrete[col]) 
        pairwise_square = cut_edge_label * edge_attr # wij*(xi-xj)^2
        
        solution_value = torch.zeros(num_graphs, device = device) 

        for graph in range(num_graphs):
            connected_label = 1
            batch_graph = (edge_batch_index==graph)
            batch_node = (batch==graph)
            solution_value[graph] = (pairwise_square[batch_graph].unsqueeze(-1)).sum()/2 
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
            judge = nx.Graph()
            judge.add_edges_from(connections)
            
            node_indices = set(judge.nodes())
            diff = node_indices_graph.difference(node_indices)
            if len(diff) != 0: 
                judge.add_nodes_from(list(diff)) 
            components = connected_components(judge) 
            if len(components) != 2:
                connected_label = 0
                not_connected += 1
            else:
                mean_sum = mean_sum + solution_value[graph]

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
            initial_graph.add_edges_from(connections_initial) #

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
            tree = kruskal(edge_tree, len(discrete[batch_node]))
            initial_tree = nx.Graph()
            initial_tree.add_edges_from(tree) 
            max_cut, best_value = heuristic(initial_graph, initial_tree) 
            mean_sum_h = mean_sum_h + best_value

        count += 1
    t1 = time.time() - t0
    t1 = t1/float(len(test_dataset))
    mean_sum = mean_sum/float(len(test_dataset))
    mean_sum_h = mean_sum_h/float(len(test_dataset))  
    print("solution not connected:" + str(not_connected)) 
    print("unsupervised value mean:" + str(mean_sum))
    print("elect value mean:" + str(mean_sum_h)) 
    print("elect time:" + str(t1))

    # output
    msg = f'solution not connected: {not_connected},\n unsupervised value mean:{mean_sum}, \n elect value mean:{mean_sum_h},\n elect time: {t1}'
    log_msg(msg, log_file)

    return mean_sum, mean_sum_h, t1, not_connected


aaa = 0
for abc in range(5):
    if __name__ == '__main__':
        mean_sum, mean_sum_h, t1, not_connected = main()
    aaa +=1

