import networkx as nx


def add_nodes(G: nx.Graph, T: nx.Graph, local: nx.Graph, add_edge: tuple, remove_edge: list):
    """ Adding Nodes operations in Heuristic Algorithm.

    Args:
        add_edge (tuple): (s, t), s in LOCAL, t in CANDIDATE
    """

    current_tree = T.copy()
    
    current_tree.add_edge(*add_edge, weight=G.get_edge_data(*add_edge))
    current_tree.remove_edges_from(remove_edge)
    
    current_local_nodes = list(local.nodes)
    current_local_nodes.append( add_edge[1] )
    current_local = current_tree.subgraph(current_local_nodes).copy()
    
    current_candidate_nodes = list( set(G.nodes) ^ set(current_local_nodes) )
    current_candidate = current_tree.subgraph(current_candidate_nodes).copy()
    
    # RECONNECT
    reconnect(G, current_tree, add_edge[1], current_local, current_candidate)
    assert nx.is_tree(current_tree), 'Error: Wrong reconnect algorithm.'
    
    return current_tree


def reconnect(G: nx.Graph, T: nx.Graph, V, local: nx.Graph, candidate: nx.Graph):
    """ Reconnect connected components in candidate.
        Use kruskal to generate MST.

    """
    
    graph_candidate = G.subgraph(candidate)
    
    candidate_forest: nx.Graph
    candidate_forest = nx.minimum_spanning_tree(graph_candidate, weight='weight', algorithm="kruskal")
    
    T.remove_edges_from(candidate.edges)
    T.add_weighted_edges_from(candidate_forest.edges(data=True), weight='weight')
    
    # some vertices only connected with local on graph
    for tree in nx.connected_components(T):
        
        if V in tree:
            continue
        
        for n in nx.edge_boundary(G, tree, local.nodes):
            
            # T
            T.add_edge(*n, weight=G.get_edge_data(*n))
            
            # local
            local.add_edge(*n, weight=G.get_edge_data(*n))
            local.add_weighted_edges_from(T.subgraph(tree).edges(data=True), weight='weight')
            
            break
        
    return
