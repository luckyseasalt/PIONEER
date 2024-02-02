#imports
import torch
from itertools import product
import time
from torch import tensor
from torch.optim import Adam
import torch.nn

from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from cut_utils import get_diracs
import GPUtil
import pickle
from torch_geometric.data import  DataLoader
from random import shuffle
from torch_geometric.datasets import TUDataset
import numpy as np

from pre_process import bridge
from models import ELECT_Mnist
from cut_utils import get_diracs

from torch.optim import Adam
from itertools import product
import GPUtil
import os
import argparse
from pathlib import Path
import sys

from modules_and_utils import log_msg

from models import ELECT_Mnist
from test import test
import networkx as nx
from test import kruskal
from torch_geometric.data import Data
import copy
from torch_geometric.utils import to_dense_adj, to_undirected
from pre_process import connected,connected_components
from brute_force import brute_force, torch_to_networkx

# Part 1 Datasets
datasets = ["IMDB-BINARY","REDDIT-BINARY","ENZYMES"]
dataset_name = datasets[2]

if dataset_name == "IMDB-BINARY":
    stored_dataset = open('datasets/IMDB-BINARY.p', 'rb')
    dataset = pickle.load(stored_dataset)
elif dataset_name == "REDDIT-BINARY":
    stored_dataset = open('datasets/REDDIT-BINARY.p', 'rb')
    dataset = pickle.load(stored_dataset)
elif dataset_name == "ENZYMES":
    stored_dataset = open('datasets/ENZYMES.p', 'rb')
    dataset = pickle.load(stored_dataset)

# Part 2 Determining Connectivity
for_del, connections = connected(dataset)
if len(for_del) != 0:
    print("data is not connected")
data_new = []
for i in range(len(dataset)):
    if i in for_del:
        continue
    else:
        data_new.append(dataset[i])
dataset = data_new

dataset_scale = 1
total_samples = int(np.floor(len(dataset)*dataset_scale))
dataset = dataset[:total_samples]

num_trainpoints = int(np.floor(0.8*len(dataset))) # 8:1:1


num_valpoints = int(np.floor(num_trainpoints/8))
num_testpoints = len(dataset) - (num_trainpoints + num_valpoints)


traindata= dataset[0:num_trainpoints]
valdata = dataset[num_trainpoints:num_trainpoints + num_valpoints]
testdata = dataset[num_trainpoints + num_valpoints:]


#set up random seeds 
torch.manual_seed(1)
np.random.seed(2)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Part 3 Graph partitioning
train_subgraph_small, train_subgraph_large, train_bridges = bridge(traindata)
val_subgraph_small, val_subgraph_large, val_bridges = bridge(valdata) 
test_subgraph_small, test_subgraph_large, test_bridges = bridge(testdata)

torch.save(train_subgraph_large, 'train_subgraph_large.p')
torch.save(val_subgraph_large, 'val_subgraph_large.p')
torch.save(test_subgraph_large, 'test_subgraph_large.p')

# Part 4 Brute force algorithm for small graphs
brute_force_small ={} 
count = 0
max_value = 0
small_count = 0
t0 = time.time()

for subgraph in test_subgraph_small:
    if count == 0:
        max_value_for_small = 0
        label = subgraph.node_feature_label
    subgraph_count = subgraph.node_feature_label
    if int(label) != int(subgraph_count):
       max_value_for_small = 0
    G = torch_to_networkx(subgraph)
    cut, value = brute_force(G)
    if value > max_value_for_small:
        max_value_for_small = value
        dict_b = {int(subgraph_count):{value:cut}}
        brute_force_small.update(dict_b)
    count += 1
t1 = time.time() - t0
print("time:\n")
print(t1)
print("value:\n")
print(max_value/(len(test_subgraph_small)-small_count))
print()

# Part 5 Train model
def valid(net, val_loader, device, receptive_field, log_file):
    net.eval()
    totalretdict = {}
    with torch.no_grad():
        count = 1
        for data in val_loader:

            data = data.to(device)
            data_prime = get_diracs(data, 1, device, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)
            data = data.to('cpu')
            data_prime = data_prime.to(device)          
            
            edge_batch_index = data_prime.batch[data_prime.edge_index[0]]

            retdict = net(data_prime, None, edge_batch_index = edge_batch_index)

            msg = f'VAL, data batch: {count}, loss: {float(retdict["loss"][0])}, f-loss: {float(retdict["f"][0].mean().squeeze())}, g-loss: {float(retdict["g"][0].mean().squeeze())}'
            log_msg(msg, log_file)

            for key,val in retdict.items():
                if "sequence" in val[1]:
                    if key in totalretdict:
                        totalretdict[key][0] += val[0].item()
                    else:
                        totalretdict[key] = [val[0].item(),val[1]]
        count += 1
            
    return totalretdict["loss"][0]

def main():

    parser = argparse.ArgumentParser(description='this is the arg parser for real world dataset') 
    parser.add_argument('--save_path', dest = 'save_path',default = 'train_files') 
    parser.add_argument('--datasets', dest = 'datasets',default = 'ENZYMES') 
    parser.add_argument('--gpu', dest = 'gpu',default = '3') 
    parser.add_argument('--tag', default='date', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
    args = parser.parse_args()

    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date

    log_file = 'log/{}_gpu{}_synthetic dataset_{}.txt'.format(args.tag, args.gpu, args.datasets)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    train_subgraph_large = torch.load('train_subgraph_large.p')
    val_subgraph_large = torch.load('val_subgraph_large.p')
    
    # Renumber
    for data in train_subgraph_large:
        
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

    for data in val_subgraph_large:
        
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

    local_date = time.strftime('%m.%d', time.localtime(time.time()))
    log_file = 'log/{}_gpu{}_{}.txt'.format(local_date, args.gpu, dataset_name)

    #number of propagation layers
    numlayers = 4
    #size of receptive field
    receptive_field = numlayers + 1

    net =  ELECT_Mnist(dataset,numlayers, 32, 32,1) 
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    lr_decay_step_size = 5
    lr_decay_factor = 0.95

    net.to(device).reset_parameters()
    optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.00)

    b_sizes = [32]
    l_rates = [0.001]
    depths = [4]
    coefficients_f = [1.] 
    coefficients_g = [2.]
    rand_seeds = [66] 
    widths = [32] 

    epochs = 100
    retdict = {}


    for batch_size, learning_rate, numlayers, penalty_coeff_f, penalty_coeff_g, r_seed, hidden_1 in product(b_sizes, l_rates, depths, coefficients_f, coefficients_g, rand_seeds, widths):
   
        torch.manual_seed(r_seed)

        train_loader = DataLoader(train_subgraph_large, batch_size, shuffle=True)
        val_loader =  DataLoader(val_subgraph_large, batch_size, shuffle=False)

        receptive_field= numlayers + 1

        #hidden_1 = 128
        hidden_2 = 1

        net =  ELECT_Mnist(train_subgraph_large,numlayers, hidden_1, hidden_2 ,1)
        net.to(device).reset_parameters()
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.00000)

        best_train_net = 100000
        best_val_net = 100000
        stand = 0.

        for epoch in range(epochs):

            msg = f'TRAIN, epoch: {epoch} start.'
            log_msg(msg, log_file)

            totalretdict = {}
            count=0

            #learning rate schedule
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

            #show currrent epoch and GPU utilizationss
            print('Epoch: ', epoch)
            GPUtil.showUtilization()

            net.train()
            for data in train_loader:

                # renumber
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

                count += 1
                
                data = data.to(device)
                data_prime = get_diracs(data, 1, device, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)

                data = data.to('cpu')
                data_prime = data_prime.to(device)

                
                # batch index for edge
                edge_batch_index = data_prime.batch[data_prime.edge_index[0]]

                retdict = net(data_prime, None, penalty_coeff_f, penalty_coeff_g, edge_batch_index)

                # output
                msg = f'TRAIN, data batch: {count}, loss: {float(retdict["loss"][0])}, f-loss: {float(retdict["f"][0].mean().squeeze())}, g-loss: {float(retdict["g"][0].mean().squeeze())}, gamma: {float(retdict["gamma"][0])}'
                log_msg(msg, log_file)

                for key,val in retdict.items():
                    if "sequence" in val[1]:
                        if key in totalretdict:
                            totalretdict[key][0] += val[0].item()
                        else:
                            totalretdict[key] = [val[0].item(),val[1]]
                    if "gamma" in val[1]:
                        if key in totalretdict:
                            totalretdict[key][0] += val[0].item()
                        else:
                            totalretdict[key] = [val[0].item(),val[1]]
                if epoch > 2:
                    optimizer.zero_grad()
                    retdict["loss"][0].backward()

                    torch.nn.utils.clip_grad_norm_(net.parameters(),1)
                    optimizer.step()
                    del(retdict)
            loss_val = valid(net, val_loader, device, receptive_field, log_file)
            loss_val = loss_val/(len(val_loader.dataset)/batch_size)
            if epoch > -1:
                for key,val in totalretdict.items():
                    if "sequence" in val[1]:
                        val[0] = val[0]/(len(train_loader.dataset)/batch_size)
                    if "gamma" in val[1]:
                        val[0] = val[0]/(len(train_loader.dataset)/batch_size)

                # output
                msg = f'TRAIN, epoch: {epoch} end, Total train loss: {totalretdict["loss"][0]}, total val loss: {loss_val}, Total train gamma: {totalretdict["gamma"][0]}'
                log_msg(msg, log_file)
                        

                if (loss_val < best_val_net):
                    best_val_net = loss_val
                    best_test_path = os.path.join(args.save_path,'best_val_model_{}_for_{}.pth'.format(args.tag, args.datasets))
                    torch.save(net.state_dict(), best_test_path)
                if (float(totalretdict["gamma"][0]) < best_train_net):
                    best_train_net = val[0]
                    best_train_path = os.path.join(args.save_path,'best_train_model_{}_for_{}.pth'.format(args.tag, args.datasets))
                    torch.save(net.state_dict(), best_train_path)
                if (epoch%40==1 or epoch == epochs-1):
                    PATH = os.path.join(args.save_path,'epoch'+str(epoch)+'_for_'+str(args.datasets)+'.pth')
                    torch.save(net.state_dict(), PATH)
                del data_prime
        # log & print the best_acc
        msg = f'\n\n * BEST_TRAIN_LOSS: {best_train_net}\n * BEST_VAL_LOSS: {best_val_net}\n'
        log_msg(msg, log_file)


if __name__ == '__main__':
    main()

# Part 5 test and heuristic
test_subgraph_large = torch.load('test_subgraph_large.p')
for data in test_subgraph_large:
    # renumber
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

for abc in range(10):
    ELECT_large = test(test_subgraph_large)

print("End")      