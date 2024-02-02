import networkx as nx

from swap_nodes import add_nodes

# storage all value through cut
tree_value = dict()


def greedy(candidate: nx.Graph, vertex):
    """ A greedy funtion for selecting candidate nodes.

    """
    
    return nx.cut_size(candidate, tuple(vertex), weight='weight')


def max_tree_cut(G: nx.Graph, T: nx.Graph, max_value=None):
    """ maximum cut (constraint: cut only 1 edge in the tree)

    """
    
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


def heuristic(G: nx.Graph, best_tree: nx.Graph, k=None):
    """ A heuristic algorithms through Adding Nodes operations.

    """
    
    # only 2 nodes (only 1 cut)
    if G.order() == 2:
        for n in G.nodes:
            return [n]
    
    # # minimum spanning tree of subgraph
    # best_tree = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')
    
    # maximum tree cut
    local_nodes, candidate_nodes, cut_edge, best_value = max_tree_cut(G, best_tree)    
    
    local: nx.Graph
    candidate: nx.Graph
    local = best_tree.subgraph(local_nodes).copy()
    candidate = best_tree.subgraph(candidate_nodes).copy()
    
    max_cut, max_cut_else = local_nodes, candidate_nodes
    max_edge = cut_edge
    max_tree = best_tree.copy()
    max_value = best_value
    
    
    # MAIN loop
    
    Iter = 0
    # print(f'Iter {Iter}:')
    # print(best_value)
    
    # whether remove nodes from LOCAL
    rm = False
    
    # TODO: StopIterations
    while True:
        
        # neighbors of local in candidate
        # TODO: k neighbors
        neighs = dict()
        for s in local.nodes:
            
            # whether s has neighbors in candidate
            s_c = list(candidate.nodes)
            s_c.append(s)
            graph_tmp = G.subgraph(s_c)
            if not nx.is_connected(graph_tmp):
                continue
            
            neighs[s] = list()
            
            # t: neighbor of s in candidate
            for t in graph_tmp.neighbors(s):
                neighs[s].append(t)
            
            
        # s: local vertices
        for s in neighs:
            
            # t: candidate vertices
            for t in neighs[s]:
                # the neighbors of t
                t_neighs = [ (t, n) for n in candidate[t] ]
                
                # choose remove edges
                if s in cut_edge and t in cut_edge:  # (s, t) is cut-edge
                    t_neighs.pop()
                    if not t_neighs:  # same tree
                        continue
                elif t in cut_edge:  # only t \in cut-edge
                    t_neighs.pop()
                    t_neighs.append(cut_edge)
                    
                # ADD_NODES
                current_tree = add_nodes(G, best_tree, local, add_edge=(s, t), remove_edge=t_neighs)
                
                # current max tree cut
                current_cut, current_cut_else, current_cut_edge, current_value = max_tree_cut(G, current_tree)
                
                string = "add"
                if rm:
                    string = "remove"
                # print(f" {string}: {current_value}")
                
                
                if current_value <= max_value:
                    continue
                else:
                    max_cut, max_cut_else = current_cut, current_cut_else
                    max_edge = current_cut_edge
                    max_tree = current_tree.copy()
                    max_value = current_value
                    
        # remove nodes is done
        if rm:
            if max_value > best_value:
                best_value = max_value
            break
        
        # not change best value
        if max_value == best_value:
            # print("add nodes finished!\n")
            
            rm = True
            
            # swap local and candidate
            local, candidate = candidate, local
            
            # the number of node in candidate should > 1
            # also make sure cut_else != empty
            if candidate.number_of_nodes() > 1:
                continue
            else:
                break
        
        # change best
        cut_edge = max_edge
        best_tree = max_tree.copy()
        best_value = max_value
        
        local = best_tree.subgraph(max_cut).copy()
        candidate = best_tree.subgraph(max_cut_else).copy()
        
        Iter += 1
        # print('Iter ' + str(Iter) + ':')
        # print(best_value)
        # print()
        
        
    return max_cut, best_value
