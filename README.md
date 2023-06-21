```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from torchvision.datasets import CIFAR100
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
%matplotlib inline

torch.manual_seed(321)
# dataset.npz.npy -> dataset을 40000 : 10000개로 나누어놓은 데이터 셋
cifar = np.load('/content/drive/MyDrive/빅데이터분석/dataset.npz.npy', allow_pickle=True).tolist()

x_train = cifar['x_train']
y_train = cifar['y_train']
x_val = cifar['x_val']
y_val = cifar['y_val']

# dataset = np.load('/content/drive/MyDrive/빅데이터분석/cifar100_noisy_train.npz', allow_pickle=True)
# x_train = dataset['data']
# y_train = dataset['target']

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(48, padding=4),
    transforms.RandomHorizontalFlip(),
      
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

class Cifar(Dataset):
    def __init__(self, x, y, train=True, transform=None):
        self.x = x
        self.y = y
        self.transform = None
        if transform is not None:
          self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
      if self.train:
        x = torch.from_numpy(self.x[idx])
      else:
        x = torch.FloatTensor(self.x[idx])

      if self.transform is not None:
        x = self.transform(x)
      y = torch.LongTensor([self.y[idx]])
      return x, y
   
train_dataset = Cifar(x_train, y_train, transform=train_transform)
val_dataset = Cifar(x_val, y_val, train=False)

train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
vaild_loader = DataLoader(val_dataset, batch_size = 128, shuffle = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = models.resnext101_32x8d(weights= 'ResNeXt101_32X8D_Weights.IMAGENET1K_V2')
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(2048,1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024,100)
    )


model = model.to(device)

from torch.optim import *
criterion = nn.CrossEntropyLoss().to(device)

opti = AdamW(model.parameters(), lr=2e-4, weight_decay = 2e-2)
scheduler = lr_scheduler.StepLR(opti, step_size=10, gamma=0.8)

def train(model, dataloader, criterion, data_len, opti, scheduler):
  correct = 0
  model.train()
  for data, target in dataloader:
    data = data.to(device)
    target = target.view(-1).to(device)

    output = model(data)
    loss = criterion(output, target)

    opti.zero_grad()
    loss.backward()
    opti.step()
    scheduler.step()
    
    pred = output.max(1, keepdim=True)[1]        
    correct += pred.eq(target.view_as(pred)).sum().item()    
     
  acc = 100 * correct / data_len
  return acc

def evaluate(model, dataloader, criterion, data_len):
  correct = 0
  model.eval()
  for data, target in dataloader:
    data = data.to(device)
    target = target.view(-1).to(device)

    output = model(data)
    
    pred = output.max(1, keepdim=True)[1]        
    correct += pred.eq(target.view_as(pred)).sum().item()    
     
  acc = 100 * correct / data_len
  return acc

epoch = 200


for i in range(epoch):
  train_acc = train(model, train_loader, criterion, len(train_loader.dataset), opti, scheduler)
  val_acc = evaluate(model, vaild_loader, criterion, len(vaild_loader.dataset))
  if (i+1) % 5 == 0:
    torch.save(model, f'./drive/MyDrive/빅데이터분석/cifar_resnext_model.pt')
  print(f"[Epoch: {i:2d}], [Train Acc: {train_acc:3.4f}], [Val Acc: {val_acc:3.4f}]")

# Single model predict
dataset = np.load(f'/content/drive/MyDrive/빅데이터분석/cifar100_noisy_test_public.npz')['data']
test_loader = DataLoader(torch.FloatTensor(dataset), batch_size = 128, shuffle = False)

save_dir = f'/content/drive/MyDrive/빅데이터분석/cifar_resnext_model.pt'
model = torch.load(save_dir)
model = model.to(device)
result = predict(model, test_loader, len(test_loader.dataset))

def predict(model, dataloader, data_len):
  result = []
  model.eval()
  for data in dataloader:
    data = data.to(device)
    output = model(data)
    
    pred = nn.Softmax()(output)
    pred = pred.max(1, keepdim=False)[1]
    result += pred.tolist()
  return result

import pandas as pd

df = pd.DataFrame({
    "Id" : np.arange(len(dataset)).astype(int),
    "Category" : result
})
df.to_csv("my_submission.csv", index=False)
```
