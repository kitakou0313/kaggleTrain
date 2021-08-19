import torch.utils.data
import torch.nn as nn
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchinfo import summary
import numpy as np
import os

from pytorchcv.pytorchcv import train, plot_results, display_dataset, train_long, check_image_dir

import zipfile
if not os.path.exists('data/PetImages'):
    with zipfile.ZipFile('data/kagglecatsanddogs_3367a.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

check_image_dir('data/PetImages/Cat/*.jpg')
check_image_dir('data/PetImages/Dog/*.jpg')

std_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    std_normalize
])

dataset = torchvision.datasets.ImageFolder("data/PetImages", transform=trans)
trainset, testset = torch.utils.data.random_split(dataset, [20000, len(dataset) - 20000])

vgg = torchvision.models.vgg16(pretrained=True)

vgg.classifier = torch.nn.Linear(25088, 2)

for x in vgg.features.parameters():
    x.requires_grad = False

summary(vgg, (1,3,244,244))

trainset, testset = torch.utils.data.random_split(dataset, [20000, len(dataset) - 20000])
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16)
test_loader  = torch.utils.data.DataLoader(testset, batch_size=16)

train_long(vgg, train_loader, test_loader, loss_fn=nn.CrossEntropyLoss(), epochs=1, print_freq=90)

torch.save(vgg,'data/cats_dogs.pth')