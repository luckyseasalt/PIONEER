import torch
from pathlib import Path
import yaml 

import numpy as np

from torch_geometric.data import DataLoader
import argparse
import shutil

import os,inspect
current_dir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
import sys
sys.path.append("../")

from build_dataset.build_data import Synthetic_Mnist_Dataset

from models import ELECT_Mnist

from itertools import product
import time
from torch.optim import Adam
import torch.nn
# matplotlib inline
import torch
from cut_utils import get_diracs
import GPUtil
from torch_geometric.data import DataLoader
import numpy as np
from models import ELECT_Mnist
from modules_and_utils import log_msg

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

    parser = argparse.ArgumentParser(description='this is the arg parser for synthetic dataset') 
    parser.add_argument('--save_path', dest = 'save_path',default = 'train_files') 
    parser.add_argument('--gpu', dest = 'gpu',default = '5') # gpu
    parser.add_argument('--tag', default='date', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
    args = parser.parse_args()

    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date

    log_file = 'log/{}_gpu{}_synthetic dataset.txt'.format(args.tag, args.gpu)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if not os.path.exists(args.save_path): 
        os.mkdir(args.save_path) 

    # save the model and config for this training
    old_model_path = r'./models.py' 
    new_model_path = os.path.join(args.save_path,'models.py')
    shutil.copyfile(old_model_path,new_model_path)

    old_config_path = r'../build_dataset/configs/config.yaml'
    new_config_path = os.path.join(args.save_path,'config.yaml')
    shutil.copyfile(old_config_path,new_config_path)

    cfg = Path("../build_dataset/configs/config.yaml") 
    cfg_dict = yaml.safe_load(cfg.open('r'))
    dataset = Synthetic_Mnist_Dataset(cfg_dict['data']) 
    data_splits = dataset.get_idx_split()
    train_dataset = dataset[data_splits['train']]
    val_dataset = dataset[data_splits['valid']]
    test_dataset = dataset[data_splits['test']]

    #set up random seeds 
    torch.manual_seed(1)
    np.random.seed(2)   # 设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #number of propagation layers 
    numlayers = 4
    #size of receptive field 
    receptive_field = numlayers + 1

    net =  ELECT_Mnist(dataset,numlayers, 32, 32,1) 
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    lr_decay_step_size = 5
    lr_decay_factor = 0.90

    net.to(device).reset_parameters()
    optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.00)


    b_sizes = [32] 
    l_rates = [0.001] 
    depths = [4] 
    coefficients = [2] 
    rand_seeds = [66] 
    widths = [32] 

    epochs = 100
    retdict = {}


    for batch_size, learning_rate, numlayers, penalty_coeff, r_seed, hidden_1 in product(b_sizes, l_rates, depths, coefficients, rand_seeds, widths):
   
        torch.manual_seed(r_seed)


        train_loader = DataLoader(train_dataset, batch_size, shuffle=True) 
        val_loader =  DataLoader(val_dataset, batch_size, shuffle=False)

        receptive_field= numlayers + 1

        #hidden_1 = 128
        hidden_2 = 1

        net =  ELECT_Mnist(dataset,numlayers, hidden_1, hidden_2 ,1)
        net.to(device).reset_parameters()
        optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.00000)

        best_train_net = 100000
        best_val_net = 100000
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
                count += 1
                optimizer.zero_grad(), 
                data = data.to(device)
                data_prime = get_diracs(data, 1, device, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)

                data = data.to('cpu')
                data_prime = data_prime.to(device)

                edge_batch_index = data_prime.batch[data_prime.edge_index[0]]

                retdict = net(data_prime, None, penalty_coeff, edge_batch_index)
                
                # output
                msg = f'TRAIN, data batch: {count}, loss: {float(retdict["loss"][0])}, f-loss: {float(retdict["f"][0].mean().squeeze())}, g-loss: {float(retdict["g"][0].mean().squeeze())}'
                log_msg(msg, log_file)

                for key,val in retdict.items():
                    if "sequence" in val[1]:
                        if key in totalretdict:
                            totalretdict[key][0] += val[0].item()
                        else:
                            totalretdict[key] = [val[0].item(),val[1]]

                if epoch > 2:
                    retdict["loss"][0].backward()
                    #reporter.report()

                    torch.nn.utils.clip_grad_norm_(net.parameters(),1)
                    optimizer.step()
                    del(retdict)
            loss_val = valid(net, val_loader, device, receptive_field, log_file)
            if epoch > -1:
                for key,val in totalretdict.items():
                    if "sequence" in val[1]:
                        val[0] = val[0]/(len(train_loader.dataset)/batch_size)
                        loss_val = loss_val/(len(val_loader.dataset)/batch_size)

                        # output
                        msg = f'TRAIN, epoch: {epoch} end. Total train loss: {val[0]}'
                        log_msg(msg, log_file)

                        if (loss_val < best_val_net):
                            best_val_net = loss_val
                            best_test_path = os.path.join(args.save_path,'best_val_model.pth')
                            torch.save(net.state_dict(), best_test_path)
                        if (val[0] < best_train_net):
                            best_train_net = val[0]
                            best_train_path = os.path.join(args.save_path,'best_train_model.pth')
                            torch.save(net.state_dict(), best_train_path)
                        if (epoch%20==1 or epoch == epochs-1):
                            PATH = os.path.join(args.save_path,'epoch'+str(epoch)+'.pth')
                            torch.save(net.state_dict(), PATH)
                del data_prime
        # log & print the best_acc
        msg = f'\n\n * BEST_TRAIN_LOSS: {best_train_net}\n'
        log_msg(msg, log_file)

            

if __name__ == '__main__':
    main()