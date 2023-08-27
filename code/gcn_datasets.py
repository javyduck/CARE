import setGPU
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import datetime
from time import time
from math import ceil
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T
from architectures import get_architecture, GCN
from torch_geometric.loader import DataLoader
from datasets import get_dataset
from train_utils import setup_seed
from torch.nn import  Sigmoid, Softmax
import argparse

### the dataset for GCN training ###
def get_gcn_dataset(dataset: str, noise_sd :float, split: str) -> Dataset:
    """Return the dataset as a PyTorch Dataset object
    -1 means we do not train that knowledge predictor"""

    if dataset == "AWA":
        return AwAGCNDataset(noise_sd = noise_sd, split = split)
    elif dataset == "word50":
        return WordGCNDataset(noise_sd = noise_sd, split = split)
    elif dataset == "stop_sign":
        return StopGCN(noise_sd = noise_sd, split = split)
    
class AwAGCNDataset(Dataset):
    def __init__(self, 
                 root='AWA', 
                 noise_sd=0.25, 
                 split='train', 
                 update_batch=1000, 
                 transform=None, 
                 main_num=50, 
                 attribute_num=85, 
                 hierarchy_num=28, 
                 gt_matrix_path='../data/Animals_with_Attributes2/gt_matrix.pt'):
        super(AwAGCNDataset, self).__init__()

        self.root = root
        self.noise_sd = noise_sd
        self.split = split
        self.update_batch = update_batch
        self.transform = transform

        self.main_num = main_num
        self.attribute_num = attribute_num
        self.hierarchy_num = hierarchy_num
        self.total_num = self.main_num + self.attribute_num + self.hierarchy_num
        self.gt_matrix = torch.load(gt_matrix_path)
        
        self.dataset = self.get_dataset(self.root, self.split)
        self.x = self.update_x()
        self.main_label = self.dataset.targets

        self.rule = torch.cat((torch.eye(self.main_num), self.gt_matrix), dim=1)
        self.y = self.rule[self.main_label]
        self.edge_index = self.create_edge_index()

        self.formula, self.bias = self.get_w()

    def update_x(self):
        print('Add noise in x...')
        ### path for saving main model and attribute models ###
        all_path=[f'saved_models/AwA/noise_sd_{self.noise_sd:.2f}/main.pth.tar']
        all_path.extend([f'saved_models/AwA/noise_sd_{self.noise_sd:.2f}/attribute_{i}.pth.tar' for i in range(self.attribute_num)])
        all_path.extend([f'saved_models/AwA/noise_sd_{self.noise_sd:.2f}/hierarchy_{i}.pth.tar' for i in range(self.hierarchy_num)])
                            
        x = torch.FloatTensor()
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.update_batch, shuffle=False)
        
        for batch, _ in loader:
            batch += torch.randn_like(batch) * self.noise_sd
            for model_id, path in enumerate(all_path):
                checkpoint = torch.load(path)
                if model_id == 0:
                    model = get_architecture('resnet50', 'AWA', classes = self.main_num)
                    m = Softmax(dim=1)
                else:
                    model = get_architecture('resnet50', 'AWA', classes = 1)
                    m = Sigmoid(dim=1)
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()
                with torch.no_grad():
                    confidence = torch.FloatTensor([])
                    logits = model(batch.cuda())
                    confidence = torch.cat((confidence,m(logits).detach().cpu()),dim=1)
            x = torch.cat((x,confidence),dim=0)
        return x.unsqueeze(-1)

    def get_w(self):
        ## define the reasoning matrix and bias
        mat = torch.FloatTensor()
        bias = []
        for i in range(len(self.rule)):
            tmp = self.rule[i]
            for j in torch.nonzero(tmp[:-self.hierarchy_num]).squeeze():
                init = torch.zeros(1, self.total_num)
                init[0, i] = 1
                init[0, self.attribute_num + j] = -1
                mat = torch.cat((mat, init.clone()), dim=0)
                bias.append(0)
        for i in range(self.hierarchy_num):
            tmp = torch.nonzero(self.rule[:, -self.hierarchy_num + i]).squeeze()
            init = torch.zeros(1, self.total_num)
            init[0, -self.hierarchy_num + i] = 1
            init[0, tmp] = -1
            mat = torch.cat((mat, init.clone()), dim=0)
            bias.append(0)
        return mat.T.cuda(), torch.FloatTensor(bias).cuda()

    def create_edge_index(self):
        ### define the edge in the graph
        row_1 = []
        row_2 = []
        for i in range(self.main_num):
            cur_y = self.main_num + torch.nonzero(self.gt_matrix[i]).squeeze()
            cur_x = [i] * cur_y.shape[0]
            row_1.extend(cur_x+cur_y)
            row_2.extend(cur_y+cur_x)
        row_1 = torch.tensor(row_1).unsqueeze(0).long()
        row_2 = torch.tensor(row_2).unsqueeze(0).long()
        edge_index = torch.cat((row_1, row_2),dim=0)
        return edge_index
    
    def __len__(self):
        return len(self.x)

    def get(self, idx):
        data = Data(x=self.x[idx], y = self.y[idx], edge_index=self.edge_index)
        return data


class StopGCNDataset(Dataset):
    def __init__(self, root='stop_sign', noise_sd=0.25, split='train', transform=None, gt_matrix_path='../data/stop_sign/stop_sign_gt_matrix.pt'):
        super(StopGCNDataset, self).__init__()
        self.root = root
        self.noise_sd = noise_sd
        self.split = split
        self.transform = transform
        self.gt_matrix = torch.load(gt_matrix_path)
        
        self.dataset = get_dataset(root, split)
        self.x = self.update_x()
        print("Add noise Done..")
        self.main_label = self.dataset.label
        self.main_num = len(self.gt_matrix)  # moved to instance variable
        self.rule = torch.cat((torch.eye(self.main_num), self.gt_matrix), dim=1)
        self.total_num = self.rule.shape[1]
        self.y = self.rule[self.main_label]
        self.edge_index = self.create_edge_index()

    def update_x(self, batch_size=5000):
        # Moved noise_sd and other args into the method
        print('Add noise in x...')
        all_path = [f'saved_models/stop_sign/noise_sd_{self.noise_sd:.2f}/{self.root}.pth.tar']
        all_path.extend([f'saved_models/stop_sign/noise_sd_{self.noise_sd:.2f}/attribute_{i}.pth.tar' for i in range(20)])
        
        assemble = torch.FloatTensor()
        self.dataset.data += torch.randn_like(self.dataset.data) * self.noise_sd
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        
        for model_id, path in enumerate(all_path):
            checkpoint = torch.load(path)
            if model_id == 0:
                model = get_architecture('neural', 'stop_sign')
            else:
                model = get_architecture('neural_attribute', 'stop_sign')
            m = Softmax(dim=1)
                
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            with torch.no_grad():
                confidence = torch.FloatTensor()
                for batch, _ in loader:
                    batch = batch.cuda()
                    logits = model(batch)
                    confidence = torch.cat((confidence, m(logits).detach().cpu()[:, [1 if model_id != 0 else slice(None)]]), dim=0)
            assemble = torch.cat((assemble, confidence), dim=1)
        return assemble.unsqueeze(-1)
    
    def create_edge_index(self):
        # Create edge index logic here
        ## create edge_index
        row_1 = []
        row_2 = []
        for i in range(12):
            cur_y = 12 + torch.nonzero(self.gt_matrix[i]).reshape(-1)
            cur_x = [i] * cur_y.shape[0]
            row_1.extend(cur_x)
            row_1.extend(cur_y)
            row_2.extend(cur_y)
            row_2.extend(cur_x)
            
        hierarchy_1 = 12 + torch.tensor([4,5,6,12,13,2,6,7,8,9,0,1,2,3])
        hierarchy_2 = 12 + torch.tensor([14,14,14,14,14,15,15,16,16,16,19,19,19,19])
    
        row_1.extend(hierarchy_1)
        row_2.extend(hierarchy_2)
            
        row_1 = torch.tensor(row_1).unsqueeze(0).long()
        row_2 = torch.tensor(row_2).unsqueeze(0).long()
        
        self.edge_index = torch.cat((row_1,row_2),dim=0)
        
    def get_w(self):
        mat = torch.FloatTensor()
        bias = []
        for i in range(len(self.gt_matrix)):
            tmp = gt_matrix[i]
            q = 12 + torch.nonzero(tmp).reshape(-1)
            for j in q:
                init = torch.zeros(1,num_f)
                if j < 12+12:
                    init[0,i] = -1
                    init[0,j] = 1
                else:
                    init[0,i] = 1
                    init[0,j] = -1
                mat = torch.cat((mat,init.clone()),dim=0)
                bias.append(0)

        hierarchy_1 = 12 + torch.tensor([4,5,6,12,13,2,6,7,8,9,0,1,2,3])
        hierarchy_2 = 12 + torch.tensor([14,14,14,14,14,15,15,16,16,16,19,19,19,19])

        for j in range(len(hierarchy_1)):
            init = torch.zeros(1,num_f)
            init[0,hierarchy_1[j]] = 1
            init[0,hierarchy_2[j]] = -1
            mat = torch.cat((mat,init.clone()),dim=0)
            bias.append(0)
        return mat.T.cuda(), torch.FloatTensor(bias).cuda()
    
    def __len__(self):  # Renamed from len
        return len(self.x)

    def get(self, idx):
        data = Data(x=self.x[idx], y = self.y[idx], edge_index=self.edge_index)
        return data
    
    
class WordGCNDataset(Dataset):
    def __init__(self, root='word50_word', noise_sd=0.12, split='train', transform=None, gt_matrix_path='../data/word50/word50_gt_matrix.pt'):
        super(WordGCNDataset, self).__init__()
        self.root = root
        self.noise_sd = noise_sd
        self.split = split
        self.transform = transform
        self.dataset = get_dataset(root, split)
        self.x = self.update_x()
        self.main_label = self.dataset.label
        
        # Load the ground truth matrix (gt_matrix)
        gt_matrix = torch.load(gt_matrix_path)
        gt_matrix = {value: eval(key) for key, value in gt_matrix.items()}
        for i in range(50):
            for j in range(5):
                gt_matrix[i][j] += 50 + 26 * j
            gt_matrix[i].insert(0, i)
        self.gt_matrix = gt_matrix
        
        self.main2all = list(map(lambda x: gt_matrix[x], torch.arange(50).tolist())) 
        y = torch.zeros(len(self.main2all), self.x.shape[1])
        y[torch.arange(50).reshape(-1, 1), self.main2all] = 1
        self.rule = y.long()
        self.total_num = self.rule.shape[1]
        self.y = self.rule[self.main_label]
        
        self.edge_index = self.create_edge_index()  # Assuming you have this function defined

    def update_x(self, batch_size = 2000):
        print('Add noise in x...')
        data = self.dataset.data.clone().reshape(-1,5*28*28) 
        data += torch.randn_like(data) * self.noise_sd
        
        all_path=['saved_models/word50/main/main-1_noise_sd%.2f.pth.tar' % self.noise_sd]
        all_path.extend(['saved_models/word50/extra/extra-%d_noise_sd%.2f.pth.tar' % (i,self.noise_sd) for i in range(1,6)])

        assemble = torch.FloatTensor()
        
        for model_id, path in enumerate(all_path):
            if model_id == 0:
                x = data
                model = get_architecture('MLP', 'word50_word')
            else:
                x = data[:, 28 * 28 * (model_id - 1): 28 * 28 * model_id]
                model = get_architecture('MLP', 'word50_letter')
                
            loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=False)
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            m = Softmax(dim=1)
            with torch.no_grad():
                confidence = torch.FloatTensor([])
                for batch in loader:
                    logits = model(batch.cuda())
                    confidence = torch.cat((confidence,m(logits).detach().cpu()),dim=0)
            assemble = torch.cat((assemble,confidence),dim=1)
        print('Done')
        return assemble.unsqueeze(-1)

    def create_edge_index(self):
        row_1 = []
        row_2 = []
        for j in range(50):
            tmp = self.gt_matrix[j]
            for k in range(1,6):
                row_1.append(tmp[0])
                row_2.append(tmp[k])
                row_1.append(tmp[k])
                row_2.append(tmp[0])
        row_1 = torch.LongTensor(row_1).reshape(1,-1)
        row_2 = torch.LongTensor(row_2).reshape(1,-1)
        edge_index = torch.cat((row_1,row_2),dim=0)
        return edge_index
    
    def get_w(self):
        # wrapper function
        def subsets_of_given_size(numbers, n):
            return [x for x in subsets(numbers) if len(x)==n]
        numbers = [1, 2, 3, 4, 5]
        pick3 = subsets_of_given_size(numbers, 3)
        pick4 = subsets_of_given_size(numbers, 4)
        pick5 = subsets_of_given_size(numbers, 5)

        mat = torch.FloatTensor()
        bias = []

        for i in range(len(self.main2all)):
            tmp = torch.tensor(self.main2all[i])

            for j in range(1, 6):  
                init = torch.zeros(1, 180)
                init[0, tmp[0]] = 1
                init[0, tmp[j]] = -1
                mat = torch.cat((mat, init.clone()), dim=0)
                bias.append(0)

            for j in pick3:
                init = torch.zeros(1, 180)
                init[0, tmp[0]] = -1
                init[0, tmp[j]] = 1 / len(j)
                mat = torch.cat((mat, init.clone()), dim=0)
                bias.append(0)

            for j in pick4:
                init = torch.zeros(1, 180)
                init[0, tmp[0]] = -1
                init[0, tmp[j]] = 1 / len(j)
                mat = torch.cat((mat, init.clone()), dim=0)
                bias.append(0)

            for j in pick5:
                init = torch.zeros(1, 180)
                init[0, tmp[0]] = -1
                init[0, tmp[j]] = 1 / len(j)
                mat = torch.cat((mat, init.clone()), dim=0)
                bias.append(0)

        return mat.T.cuda(), torch.FloatTensor(bias).cuda()
    
    def __len__(self):
        return len(self.x)

    def get(self, idx):
        data = Data(x=self.x[idx], y = self.y[idx], edge_index=self.edge_index)
        return data