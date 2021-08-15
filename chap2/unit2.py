import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from pytorchcv import pytorchcv

pytorchcv.load_mnist()

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10),
    nn.LogSoftmax()
)

print(data_train[0][1])
print((torch.exp(net(data_train[0][0]))))

train_loader = DataLoader(data_train, batch_size=64)
test_loader = DataLoader(data_test,batch_size=64)


def trainEpoch(net:torch.nn.Sequential, dataLoader:DataLoader, lr=0.01, optimizer=None, lossFn= nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    totalLoss, acc, count = 0,0,0

    for feature, label in dataLoader:
        optimizer.zero_grad()
        out = net(feature)
        loss = lossFn(out, label)

        loss.backward()
        optimizer.step()

        totalLoss += loss
        _,predicted = torch.max(out,1)
        acc+=(predicted==label).sum()
        count+=len(label)
    return totalLoss.item()/count, acc.item()/count

trainEpoch(net, train_loader)