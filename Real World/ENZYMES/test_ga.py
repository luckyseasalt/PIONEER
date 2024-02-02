import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader, Data
import argparse

from models import ELECT_Mnist
from cut_utils import get_diracs

import random
import time

import CAL_MAX_CUT
from CAL_MAX_CUT import load
from CAL_MAX_CUT import graph_ga
from CAL_MAX_CUT import geneticalgorithm
from CAL_MAX_CUT import max_cut

import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
import sys
sys.path.append("../") 

from modules_and_utils import log_msg

# Part 1 Dataset
subgraph_large = torch.load('test_subgraph_large.p')

local_date = time.strftime('%m.%d', time.localtime(time.time()))

log_file = 'log/result_ga_{}.txt'.format(local_date)

def kruskal(edge_list, node_num):
    # renumber
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

    # Constructing spanning tree
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

    # Recovery number
    mapping_convert = {}
    
    for key, value in mapping.items():
        mapping_convert[value] = key
    count_def_2 = 0
    for i in tree_st:
        tree_st[count_def_2][0] = mapping_convert[int(i[0])]
        tree_st[count_def_2][1] = mapping_convert[int(i[1])]
        count_def_2 += 1

    return tree_st


def main(subgraph_large):

    parser = argparse.ArgumentParser(description='this is the arg parser for synthetic dataset with power grid')
    parser.add_argument('--model_path', dest = 'model_path',default = 'train_files/best_val_model_12.23_for_ENZYMES.pth')
    parser.add_argument('--gpu', dest = 'gpu',default = '3')
    args = parser.parse_args()

    batch_size = 32
    numlayers = 4

    test_loader = DataLoader(subgraph_large, batch_size, shuffle=False)
     
    for data in subgraph_large:
        
        num_nodes = len(data.x)
        
        edge_index_old = data.edge_index
        edge_index_new = edge_index_old.clone()
        old_node_labels_list = edge_index_old.reshape(1,-1).tolist()
        old_node_labels = sorted(list(set(old_node_labels_list[0])))
        
        new_node_labels = torch.arange(num_nodes)
        
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

    data_num = len(subgraph_large)
    datas_total = len(subgraph_large)
    best_cut_sum = 0
    
    t0 = time.time()
    
    for data in test_loader:

        data = data.to(device)
        data_prime = get_diracs(data.to(device), 1, device, sparse = True, effective_volume_range=0.15, receptive_field = 7)
        data_prime = data_prime.to(device)
        
        edge_index = data_prime.edge_index
        edge_attr = torch.ones(len(edge_index[0]))
        batch = data_prime.batch
        num_graphs = batch.max().item() + 1 
        row, col = edge_index 

        edge_batch_index = batch[edge_index[0]]  

        retdz = model(data_prime, edge_batch_index= edge_batch_index) 

        discrete = torch.round(retdz["output"][0]) 

        x = retdz["output"][0]
        
        cut_edge_label = torch.square(discrete[row] - discrete[col]) 
        
        s = 0
        
        for graph in range(num_graphs): 
            batch_graph = (edge_batch_index==graph)
            batch_node = (batch==graph)
            batch_graph_2 = torch.cat((batch_graph.view(1,-1),batch_graph.view(1,-1)),0)
            index_bi = edge_index[batch_graph_2].reshape(2,-1)
            cut_edge_graph_bi = cut_edge_label[batch_graph]

            # ga
            e = index_bi.T
            new_e = e.clone()
            edge = new_e - s - 1
            vertice = x[batch_node].size(0)
            s = s + vertice
            g = graph_ga.Graph(vertice, edge)
            known_cut = [random.choice([0, 1]) for _ in range(vertice)]
            ga = geneticalgorithm.GeneticAlgorithm(g, population_size=100, mutation_rate=0.05, max_generations=100, known_cut=known_cut)
            max_cuts_ga = max_cut.MaxCut(g, ga)
            max_cuts, cuts, isvalid = max_cuts_ga.result()
            best_cut_sum = best_cut_sum + max_cuts
            if isvalid == 0:
                datas_total = datas_total - 1
            
        count += 1
    t1  = (time.time() - t0)/float(data_num)
    print("avg time:")
    print(t1)
    best_cut_sum = best_cut_sum/float(datas_total)
    print("avg value:")
    print(best_cut_sum)
    print("not connected:")
    print(float(data_num)-float(datas_total))

    # 输出
    msg = f'ga value::{best_cut_sum},\n ga time: {t1},\n not connected:{float(data_num)-float(datas_total)}'
    log_msg(msg, log_file)

    return t1, best_cut_sum

for abc in range(10):
    if __name__ == '__main__':
        t1, avg_value = main(subgraph_large)