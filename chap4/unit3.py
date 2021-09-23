import enum
from typing import Optional
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import pandas as pd
import os

data_path = './data/spectrograms'
yes_no_dataset = datasets.ImageFolder(
    root=data_path,
    transform=transforms.Compose(
        [transforms.Resize((201,81)),transforms.ToTensor()]
    )
)

print(yes_no_dataset)
print(yes_no_dataset[5][0].size())

train_size = int(0.8 * len(yes_no_dataset))
test_size = len(yes_no_dataset) - train_size
yes_no_train_dataset, yes_no_test_dataset = torch.utils.data.random_split(yes_no_dataset, [train_size, test_size])

train_dataloader = torch.utils.data.DataLoader(
    yes_no_train_dataset,
    batch_size=15,
    num_workers=2,
    shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(
    yes_no_test_dataset,
    batch_size=15,
    num_workers=2,
    shuffle=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

class CNNet(nn.Module):
    """
    docstring
    """
    def __init__(self):
        """
        docstring
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3,32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136,50)
        self.fc2 = nn.Linear(50,2)

    def forward(self,x):
        """
        docstring
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x,dim=1)
model = CNNet().to(device)

cost = torch.nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(dataloader:DataLoader, model:CNNet, loss, optimizer:torch.optim.Adam):
    model.train()
    size = len(dataloader.dataset)

    for batch, (X,Y) in enumerate(dataloader):
        X,Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)

def test(dataloader:DataLoader,model:CNNet):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, corrent = 0,0

    with torch.no_grad():
        for batch, (X,Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            test_loss += cost(pred, Y).item()
            corrent += (pred.argmax(1)==Y).type(torch.float).sum().item()
    
    test_loss /= size
    corrent /= size

epoch = 15

for t in range(epoch):

    train(train_dataloader, model, cost, optimizer)
    test(test_dataloader, model)

print("Done!")
    
