from typing import List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, remove_self_loops, is_undirected, to_undirected

import sys
sys.setrecursionlimit(10000) 

def connected(dataset):
    for_del = []
    data_count = 0
    for data in dataset:
        edge_index = data.edge_index
        connections = []
        for j in range(len(edge_index[0])):
            connections.append((int(edge_index[0][j]),int(edge_index[1][j])))

        graph = nx.Graph()
        
        graph.add_edges_from(connections)
        components = connected_components(graph)
        
        if len(components) != 1:
            for_del.append(data_count)
        data_count += 1

    return for_del,connections

def bridge(dataset):

    data_label = 0 
    subgraph_small_list =[] 
    subgraph_large_list = [] 
    bridge_list = [] 
    
    for data in dataset:

        edge_index = data.edge_index

        if not is_undirected(edge_index):
                data.edge_index = to_undirected(data.edge_index)
        
        remove_edge = remove_self_loops(edge_index)
        edge_index = remove_edge[0]

        edge_index = torch.unique(edge_index, dim=1)

        connections = []
        for j in range(len(edge_index[0])):
            connections.append((int(edge_index[0][j]),int(edge_index[1][j])))

        graph_1 = nx.Graph()
        
        graph_1.add_edges_from(connections)
        
        degree_dict = dict(graph_1.degree())
        label_b = []
        for j in degree_dict: 
            if degree_dict[j] == 1:
                label_b.append(j)
        leaf = []
        for j in range(len(edge_index[0])):
            if int(edge_index[0][j]) in label_b or int(edge_index[1][j]) in label_b:
                if int(edge_index[0][j]) < int(edge_index[1][j]):
                    leaf.append((int(edge_index[0][j]),int(edge_index[1][j])))
                else:
                    leaf.append((int(edge_index[1][j]),int(edge_index[0][j])))
        graph_1.remove_nodes_from(label_b)

        # find the bridges of graph
        bridges_yield = nx.bridges(graph_1)
        bridges = []
        for by in bridges_yield:
            bridge = list(by)
            bridge.append(graph_1.get_edge_data(*bridge))
            if bridge[0] < bridge[1]:
                bridges.append((bridge[0],bridge[1]))
            else:
                bridges.append((bridge[1],bridge[0]))
        bridges = bridges + leaf
        
        node_feature_label =  torch.tensor(data_label)
        if len(bridges) != 0:
            bridge_data = Data(x = torch.tensor(bridges[0]), node_feature_label = node_feature_label)
            bridge_list.append(bridge_data)

            bridges += [(b, a) for a, b in bridges]
            graph_no_bridge = [i for i in connections if i not in bridges]
           
            graph = nx.Graph()
            
            graph.add_edges_from(graph_no_bridge)

            components = connected_components(graph)

            for subset in components:
                subset = torch.tensor(subset) 
                sub_edge_index, _ = subgraph(subset, edge_index)
                sub_node_feature = torch.zeros(len(subset))
                subgraph_data = Data(x = sub_node_feature, edge_index = sub_edge_index, node_feature_label = node_feature_label)
                if len(subset) < 16: 
                    subgraph_small_list.append(subgraph_data)
                else:
                    num_nodes = len(data.x)
                    if num_nodes != len(subset):
                    # renumber
                        edge_index_old = subgraph_data.edge_index
                        old_node_labels_list = edge_index_old.reshape(1,-1).tolist()
                        old_node_labels = sorted(list(set(old_node_labels_list[0])))
                        
                        new_node_labels = torch.arange(num_nodes)
                        
                        node_mapping = dict(zip(old_node_labels, new_node_labels.tolist()))
                        for i in range(len(edge_index_old[0])):
                            edge_index_old[0][i] = node_mapping[int(edge_index_old[0][i])]
                            edge_index_old[1][i] = node_mapping[int(edge_index_old[1][i])]
                        subgraph_data.edge_index = edge_index_old
                    subgraph_large_list.append(subgraph_data)
        else:
            num_nodes = len(data.x)
            if num_nodes < 16:
                subgraph_data = Data(x = data.x, edge_index = data.edge_index, node_feature_label = node_feature_label) 
                subgraph_small_list.append(subgraph_data)
            else:
                sub_node_feature = torch.zeros(num_nodes)
                subgraph_data = Data(x = sub_node_feature, edge_index = edge_index, node_feature_label = node_feature_label)
                subgraph_large_list.append(subgraph_data)
        data_label += 1

    return subgraph_small_list, subgraph_large_list, bridge_list
    
def connected_components(graph):
    
    components = []

    visited = []

    for node in graph.nodes:
        
        if node not in visited:
            
            component = depth_first_search(graph, node, visited)
            
            components.append(component)

    return components

def depth_first_search(graph, node, visited):
    
    component = []

    visited.append(node)

    component.append(node)

    for neighbor in graph.neighbors(node):
        if neighbor not in visited:
            component.extend(depth_first_search(graph, neighbor, visited))

    return component
