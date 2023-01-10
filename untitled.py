import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class Classifier(nn.Module):
    def __init__(self, dim_neck):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(dim_neck*2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 40)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = x[:,-1, :]
        out1 = self.fc1(out)
        out1 = self.relu(out1)
        out2 = self.fc2(out1)
        out2 = self.relu(out2)
        out3 = self.fc3(out2)
        return out3
    
class CustomDataset(Dataset):
    def __init__(self, x):
        self.x = x
        
    def __getitem__(self, index):
        data, label = self.x[index]
        
    
class Solver(object):
    def __init__(self, vcc_loader, config):
        self.dim_neck = config.dim_neck
        
        self.vcc_loader = vcc_loader
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        
        self.build_model()
        self.ce_loss = nn.CrossEntropyLoss().to(self.device)
        
    def build_model(self):
        self.C = Classifier(self.dim_neck)
        self.c_optimizer = torch.optim.Adam(self.C.parameters(), 0.0001)
        self.C.to(self.device)
        
    def c_reset_grad(self):
        self.c_optimizer.zero_grad()
        
    def train(self):
        data_loader = self.vcc_loader
        
        