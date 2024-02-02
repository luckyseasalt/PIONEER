import CAL_MAX_CUT
import numpy as np


class Graph:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.adj_matrix = self.create_adj_matrix()

    def create_adj_matrix(self):
        adj_matrix = np.zeros((self.vertices, self.vertices), dtype=int)
        for edge in self.edges:
            u, v = edge
            adj_matrix[u][v] = 1
            adj_matrix[v][u] = 1
        return adj_matrix

    def is_connected_1(self, subset):
        # 子图的顶点列表
        vertices_subset = [i for i in range(self.vertices) if subset[i] == 1]

        if not vertices_subset:
            return False  # 空子集不算连通

        # 记录每个顶点是否被访问过
        visited = [False] * self.vertices

        # 从子图的第一个顶点开始DFS
        self.dfs(subset, vertices_subset[0], visited, 1)

        # 如果子集中的所有顶点都被访问过，则子图连通
        return all(visited[v] for v in vertices_subset)

    def is_connected_0(self, subset):
        # 子图的顶点列表
        vertices_subset = [i for i in range(self.vertices) if subset[i] == 0]

        if not vertices_subset:
            return False  # 空子集不算连通

        # 记录每个顶点是否被访问过
        visited = [False] * self.vertices

        # 从子图的第一个顶点开始DFS
        self.dfs(subset, vertices_subset[0], visited, 0)

        # 如果子集中的所有顶点都被访问过，则子图连通
        return all(visited[v] for v in vertices_subset)

    def dfs(self, subset, v, visited, subset_value):
        visited[v] = True
        for i in range(self.vertices):
            if self.adj_matrix[v][i] != 0 and subset[i] == subset_value and not visited[i]:
                self.dfs(subset, i, visited, subset_value)
