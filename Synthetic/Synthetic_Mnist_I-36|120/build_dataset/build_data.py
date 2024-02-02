import numpy as np
import math
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from pathlib import Path
import yaml

from typing import List, Tuple
import networkx as nx

import random

from torch import tensor
import copy
from torch_geometric.utils import to_dense_adj, to_undirected, to_networkx

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

# test
def g_data():
    arr0 = np.zeros((1, 31))
    arr1 = np.zeros((1, 31))
    for j in range(31):
        arr1[0][j] = j+1
    edge_direc1_arr = np.concatenate((arr0, arr1))
    for i in range(31):
        arr0 = np.zeros((1, 30-i))
        arr1 = np.zeros((1, 30-i))
        for j in range(30-i):
            arr1[0][j] = j+i+2
        temp = np.concatenate((arr0+i+1, arr1))    
        edge_direc1_arr = np.concatenate((edge_direc1_arr, temp), axis=1)

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
    def getCuttingEdge(edges: List[Tuple]):
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
 
 
        cutting_edges = []
 
        def dfs(cur, prev, root):
            global_time[0] += 1
            dfn[cur], low[cur] = global_time[0], global_time[0]
 
            children_cnt = 0
            for next in link[cur]:
                if next != prev:
                    if dfn[next] == 0x7fffffff:
                        children_cnt += 1
                        dfs(next, cur, root)
 
                        low[cur] = min(low[cur], low[next])
 
                        if low[next] > dfn[cur]:
                            cutting_edges.append([cur, next] if cur < next else [next, cur])
                            break 
                    else:
                        low[cur] = min(low[cur], dfn[next])
 
        dfs(edges[0][0], None, edges[0][0])
        return cutting_edges
 
 
class Solution:
    def criticalConnections(self, connections: List[List[int]]) -> List[List[int]]:
        edges = [(a, b) for a, b in connections]
        cutting_edges = Tarjan.getCuttingEdge(edges)
        if len(cutting_edges)>0:
            res = 0 
        else:
            res = 1 
        return res

# the dataset class to generate the dataset 
class Synthetic_Mnist_Dataset(InMemoryDataset): 
    def __init__(self, config:dict):
        self.config = config
        self.data_path = Path(config['target_path'])
        self.splits = config['splits'] 
        super(Synthetic_Mnist_Dataset, self).__init__(root=self.data_path)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self): 
        return []

    @property
    def processed_file_names(self): 
        return ['synthetic_mnist.pt']

    def download(self): 
        pass

    def get_idx_split(self, split_type = None):
        data_idx = np.arange(10000)
        splits = self.splits
        train_num = int(float(splits['train'])*10000)
        valid_num = int(float(splits['valid'])*10000)
        if (split_type==None):
            train_idx = data_idx[:train_num]
            valid_idx = data_idx[train_num:train_num+valid_num]
            test_idx = data_idx[train_num+valid_num:]
        elif (split_type=='Random'):    
            shuffle_idx = np.random.shuffle(data_idx)
            train_idx = data_idx[:train_num]
            valid_idx = data_idx[train_num:train_num+valid_num]
            test_idx = data_idx[train_num+valid_num:]
        else:
            print("something went wrong in spliting the index?")        
        return {'train':torch.tensor(train_idx,dtype = torch.long),'valid':torch.tensor(valid_idx,dtype = torch.long),'test':torch.tensor(test_idx,dtype = torch.long)}

    def process(self):
        # create our synthetic dataset
        data_list = []
        # get the data of mnist
        mnist_list = torch.load('../build_dataset/raw_mnist/mnist_tensor.pt')
        
        num_node = 36 

        # data_store = torch.Tensor()

        # generate the edge index
        arr0 = np.zeros((1, num_node-1), dtype=int)
        arr1 = np.zeros((1, num_node-1), dtype=int)
        for j in range(num_node-1):
            arr1[0][j] = j+1
        edge_direc1_arr = np.concatenate((arr0, arr1))
        for i in range(num_node-1):
            arr0 = np.zeros((1, num_node-2-i), dtype=int)
            arr1 = np.zeros((1, num_node-2-i), dtype=int)
            for j in range(num_node-2-i):
                arr1[0][j] = j+i+2
            temp = np.concatenate((arr0+i+1, arr1))    
            edge_direc1_arr = np.concatenate((edge_direc1_arr, temp), axis=1)
        edge_direc1 = torch.from_numpy(edge_direc1_arr) 
        edge_direc1 = edge_direc1.long()

        num_data = 0

        for i in range(10000000):

            # generate the node feature
            node_feature_index = np.random.randint(1,60000,num_node) 
            node_feature_label = mnist_list[1][node_feature_index] 
            node_feature_label = node_feature_label.reshape(-1,1)
            node_feature_label_numpy = node_feature_label.numpy() 

            zeros = np.zeros(int(num_node*(num_node-1)/2)-120)
            ones = np.ones(120)
            edge_feature_direc1 = np.hstack((ones,zeros))
            random.shuffle(edge_feature_direc1)
                 
            connections = []
            num_count = 0
            for j in edge_feature_direc1:
                if j == 1:
                    connections.append((int(edge_direc1_arr[0][num_count]),int(edge_direc1_arr[1][num_count])))
                num_count += 1

            uf = UnionFind(num_node)
            for x, y in connections:
                uf.union(x, y)
            # count = sum(1 for i in range(num_node) if uf.fa[i] == i)
            label = 0
            res_connect = 1
            for j in range(num_node):
                if uf.fa[j] == j:
                    label = label + 1
                    if label > 1:
                        res_connect = 0 
                        break
            if res_connect == 0:
                continue

            s = Solution()
            res_bridge = s.criticalConnections(connections)
            if res_bridge == 0:
                continue

            num_data += 1 
            if num_data == 10001: 
                break
            if ((num_data-1)%1000==0):
                print(str((num_data-1))+"instances has been processed")

            edge_feature_direc1_reshape = edge_feature_direc1.reshape(1,-1) 
            edge_feature_direc1_tensor = torch.from_numpy(edge_feature_direc1_reshape)
            
            edge_index = edge_direc1
            
            mask = edge_feature_direc1_tensor.bool()
            
            edge_index_one = edge_index[:, mask[0]]

            # generate the lifted node feature with the dimension of |E|
            lift_index1 = edge_index_one[0,:]
            node_feature_label_lifted1 = node_feature_label_numpy[lift_index1] 
            lift_index2 = edge_index_one[1,:]
            node_feature_label_lifted2 = node_feature_label_numpy[lift_index2]
            # generate the label y
            edge_feature_arr = node_feature_label_lifted1 + node_feature_label_lifted2 + (node_feature_label_lifted1 * node_feature_label_lifted2)

            edge_feature = torch.from_numpy(edge_feature_arr) 
            edge_feature = edge_feature.view(-1)

            node_feature = torch.zeros(num_node)
            # data = Data(x = node_feature, y = y, edge_index = edge_index, edge_attr = edge_feature, node_feature_label = node_feature_label)
            
            data_num = 0

            connections_initial = []
            num_count_2 = 0
            for j in range(len(edge_index_one[0])):
                if int(edge_index_one[0][num_count_2]) < int(edge_index_one[1][num_count_2]):
                    temp_edge = (int(edge_index_one[0][num_count_2]),int(edge_index_one[1][num_count_2]))
                    connections_initial.append(temp_edge)
                else:
                    temp_edge = (int(edge_index_one[1][num_count_2]),int(edge_index_one[0][num_count_2]))
                    connections_initial.append(temp_edge)
                num_count_2 += 1
            connections_initial = list(set(connections_initial))
            tree = kruskal(connections_initial, len(node_feature))
            min_lamma3 = 1000
            for j in range(len(tree)):
                tree_copy = copy.deepcopy(tree)
                del tree_copy[j]
                tree_tensor = torch.tensor(list(zip(*tree_copy)))
                tree_tensor = to_undirected(tree_tensor)
                adj = to_dense_adj(tree_tensor).squeeze(0)
                
                laplacian0 = torch.diag(adj.sum(dim=-1)) - adj
                eigenvalues0 = torch.linalg.eigh(laplacian0) 
                eigenvalues0, _ = torch.sort(eigenvalues0[0]) 
                if eigenvalues0[2] < min_lamma3:
                    min_lamma3 = eigenvalues0[2]
            aaa = torch.log(tensor(0.01))/(-eigenvalues0[2])

            data_num += 1
            data = Data(x = node_feature, edge_index = edge_index_one, edge_attr = edge_feature, y = aaa) 

            #import pdb; pdb.set_trace()
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    configs = Path('./configs')
    for cfg in configs.iterdir():
        if str(cfg).startswith("configs/config"):
            cfg_dict = yaml.safe_load(cfg.open('r'))
            dataset = Synthetic_Mnist_Dataset(cfg_dict['data'])
    
    # this is to test the content in dataset:
    '''
    for i in range(10):
        print(dataset[i]['y'])
    '''