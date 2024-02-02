from typing import List, Tuple
import networkx as nx
import matplotlib.pyplot as plt

class UnionFind:
    def __init__(self, n: int):
        self.fa = [i for i in range(n)]

    def find(self, x: int) -> int:
        if self.fa[x] != x:
            self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def union(self, x: int, y: int):
        self.fa[self.find(x)] = self.find(y)

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


class Tarjan:
    
    @staticmethod
    def getCuttingPointAndCuttingEdge(edges: List[Tuple]):
        link, dfn, low = {}, {}, {}
        global_time = [0]
        for a, b in edges:
            if a not in link:
                link[a] = []
            if b not in link:
                link[b] = []
            link[a].append(b)
            link[b].append(a)
            dfn[a], dfn[b] = 0x7fffffff, 0x7fffffff
            low[a], low[b] = 0x7fffffff, 0x7fffffff
 
 
        cutting_points, cutting_edges = [], []
 
        def dfs(cur, prev, root):
            global_time[0] += 1
            dfn[cur], low[cur] = global_time[0], global_time[0]
 
            children_cnt = 0
            flag = False
            for next in link[cur]:
                if next != prev:
                    if dfn[next] == 0x7fffffff:
                        children_cnt += 1
                        dfs(next, cur, root)
 
                        if cur != root and low[next] >= dfn[cur]:
                            flag = True
                        low[cur] = min(low[cur], low[next])
 
                        if low[next] > dfn[cur]:
                            cutting_edges.append([cur, next] if cur < next else [next, cur])
                    else:
                        low[cur] = min(low[cur], dfn[next])
 
            if flag or (cur == root and children_cnt >= 2):
                cutting_points.append(cur)
 
        dfs(edges[0][0], None, edges[0][0])
        return cutting_points, cutting_edges
 
 
class Solution:
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        edges = [(a, b) for a, b in connections]
        cutting_dots, cutting_edges = Tarjan.getCuttingPointAndCuttingEdge(edges)
        return [(a, b) for a, b in cutting_edges]
    
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