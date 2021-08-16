import torch
from torch._C import set_anomaly_enabled
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np

from pytorchcv.pytorchcv import load_mnist, train, plot_results, plot_convolution, display_dataset

load_mnist(batch_size=128)

class MultiLayerCNN(nn.Module):
    """
    複数層NN
    """
    def __init__(self):
        super(MultiLayerCNN, self).__init__()

        self.conv1 = nn.Conv2d(1,10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))

        x = x.view(-1, 320)
        x = nn.functional.log_softmax(self.fc(x), dim=1)
        return x

net = MultiLayerCNN()
summary(net, input_size=(1,1,28,28))

hist = train(net, train_loader, test_loader, epochs=5)
print(hist)