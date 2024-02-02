import networkx as nx

from heuristic import heuristic
from brute_force import brute_force


def parallel_tree(G: nx.Graph, algorithm=heuristic):
    
    assert nx.is_connected(G), 'Error: Init-Graph is not connected.'
    assert G.order() > 1, 'Error: the number of nodes should > 1.'
    
    # find the bridges of graph
    bridges_yield = nx.bridges(G)
    bridges = []
    for by in bridges_yield:
        bridge = list(by)
        bridge.append(G.get_edge_data(*bridge))
        bridges.append(bridge)
    
    # max bridge-cut
    max_cut_set_list = []
    
    if bool(bridges):
        max_bridge = max(bridges, key=lambda e: e[-1]['weight'])

        # max_cut_set_list: storage cut-set(edges) and cut-value
        max_cut_set_list.append( [max_bridge, max_bridge[-1]['weight']] )
    
    # multi_graphs: disconnect the bridges of original graph
    multi_graphs = G.copy()
    multi_graphs.remove_edges_from(bridges)
    
    
    # parallel process sub-graph
    for cc in nx.connected_components(multi_graphs):
        
        # connected component has only 1 node
        if len(cc) == 1:
            continue
        
        # subgraph
        subgraph: nx.Graph
        subgraph = G.subgraph(cc).copy()
        
        # internal algorithm
        cut, cut_value = algorithm(subgraph)
        
        cut_set_list = [
            [ *e, subgraph.get_edge_data(*e) ] 
            for e in nx.edge_boundary(subgraph, cut)
        ]
        cut_set_list.append(cut_value)
        
        # max_cut_set append
        max_cut_set_list.append(cut_set_list)
    
    # # sum of cut-set weights
    # for l in max_cut_set_list:
    #     W = sum( w['weight'] for _, _, w in l )
    #     l.append( W )
    
    # connected max cut
    max_cut_set = max(max_cut_set_list, key=lambda e: e[-1])
    max_cut_value = max_cut_set.pop()
    
    graph_tmp = G.copy()
    graph_tmp.remove_edges_from(max_cut_set)
    
    max_cut = min(nx.connected_components(graph_tmp), key=len)
    
    return max_cut, max_cut_value
