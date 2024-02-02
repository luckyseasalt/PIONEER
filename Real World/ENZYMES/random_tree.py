import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import DataLoader, Data
import argparse
from collections import OrderedDict

from pre_process import connected, bridge, connected_components
from brute_force import brute_force, torch_to_networkx
from models import ELECT_Mnist
from cut_utils import get_diracs

from torch.optim import Adam
from itertools import product
import GPUtil

from torch_geometric.utils import to_dense_adj

from heuristic import heuristic
import random
import time

import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
import sys
sys.path.append("../") #是添加上一级目录

from modules_and_utils import log_msg

# Dataset
subgraph_large = torch.load('test_subgraph_large.p')

local_date = time.strftime('%m.%d', time.localtime(time.time()))

log_file = 'log/result_random_{}.txt'.format(local_date)

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

def max_tree_cut(G: nx.Graph, T: nx.Graph, max_value=None):
    """ maximum cut (constraint: cut only 1 edge in the tree)

    """

    tree_value = dict()
    
    if max_value is None:
        max_value = float('-inf')
    
    # traverse all edges of the tree
    for (u, v) in T.edges:
        
        tree = T.copy()
        tree.remove_edge(u, v)
        cut, cut_else = sorted(nx.connected_components(tree), key=len)
        
        value = tree_value.get(tuple(cut))
        if value is not None and value <= max_value:
            continue
        
        if value is None:
            value = nx.cut_size(G, cut, weight='weight')
            tree_value[tuple(cut)] = value
            if value > max_value:
                max_cut, max_cut_else = cut, cut_else
                cut_edge = (u, v)
                max_value = value
        else:
            max_cut, max_cut_else = cut, cut_else
            cut_edge = (u, v)
            max_value = value
            
    return max_cut, max_cut_else, cut_edge, max_value

def main():
    graph_label = set()
    data_num = len(subgraph_large)
    avg_value = 0
    t0 = time.time()

    # Random tree
    for data in subgraph_large:

        max_value = 0

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
            
        edge_index = data.edge_index
        edge_attr = torch.ones(len(edge_index[0]))
        
        
        connections_initial = []
        num_count_2 = 0
        weight = {}
        for j in range(len(edge_index[0])):
            if int(edge_index[0][num_count_2]) < int(edge_index[1][num_count_2]):
                temp_edge = (int(edge_index[0][num_count_2]),int(edge_index[1][num_count_2]))
                connections_initial.append(temp_edge)
                weight[temp_edge]=float(edge_attr[j])
            else:
                temp_edge = (int(edge_index[1][num_count_2]),int(edge_index[0][num_count_2]))
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
        
        for num in range(500):
            edge_tree = connections_initial
            random.shuffle(edge_tree)
            tree = kruskal(edge_tree, num_nodes)
            initial_tree = nx.Graph()
            initial_tree.add_edges_from(tree)
            # maximum tree cut
            _, _, cut_edge, best_value = max_tree_cut(initial_graph, initial_tree)
            if best_value > max_value:
                max_value = best_value
        avg_value = avg_value + max_value

    t1  = (time.time() - t0)/float(data_num)
    print("avg time:")
    print(t1)
    avg_value = avg_value/float(data_num)
    print("avg value:")
    print(avg_value)

    # output
    msg = f'random value::{avg_value},\n random time: {t1}'
    log_msg(msg, log_file)

    return t1, avg_value

for abc in range(10):
    if __name__ == '__main__':
        t1, avg_value = main()
    