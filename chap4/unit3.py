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