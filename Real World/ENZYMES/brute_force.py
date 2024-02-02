import networkx as nx
from itertools import chain, combinations
import torch
from torch_geometric.data import Data

def torch_to_networkx(data: Data):
    edge_index = data.edge_index.cpu().numpy()
    edge_weight = data.edge_attr.cpu().numpy() if data.edge_attr is not None else None
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        weight = edge_weight[i] if edge_weight is not None else 1
        G.add_edge(u, v, weight=weight)
    return G


def subset(nodes: tuple):
    s = list(nodes)
    return chain.from_iterable( combinations(s, t) for t in range(len(s) + 1) )

def brute_force(G: nx.Graph):
    """ A brute force implementation.

    """
    n = 0
    max_value = float('-inf')
    num_nodes = G.order()
    max_cut = []
    for nodes in subset(G.nodes):
        n += 1
        if bool(nodes) and len(nodes) <= num_nodes // 2:
            other_nodes = tuple( set(nodes) ^ set(G.nodes) )
            if nx.is_connected(G.subgraph(nodes)) and nx.is_connected(G.subgraph(other_nodes)):
                value = nx.cut_size(G, nodes, weight='weight')
                if value > max_value:
                    max_value = value
                    max_cut = [nodes,other_nodes]
                    
    return max_cut, max_value
