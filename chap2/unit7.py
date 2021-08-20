import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchinfo import summary
import os

from pytorchcv.pytorchcv import train, display_dataset, train_long, load_cats_dogs_dataset, validate, common_transform

dataset, train_loader, test_loader = load_cats_dogs_dataset()

model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=False)
model.eval()
print(model)