import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary

from pytorchcv.pytorchcv import load_mnist, train, plot_results

load_mnist(batch_size=128)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.LogSoftmax(dim=0)
)

summary(net, input_size=(1,28,28))

hist = train(net, train_loader, test_loader, epochs=5)

print(hist)