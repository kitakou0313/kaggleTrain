import torch
import torchvision
import numpy as np
from torchvision.transforms import ToTensor

dataTrain = torchvision.datasets.MNIST(
    "data",
    download=True,
    train=True,
    transform=ToTensor()
)

dataTest = torchvision.datasets.MNIST(
    "data",
    download=True,
    train=False,
    transform=ToTensor()
)