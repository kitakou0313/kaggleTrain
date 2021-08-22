import torch
import torchtext
import os
import collections
os.makedirs('./data',exist_ok=True)
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data', split=('train', 'test'))
classes = ['World', 'Sports', 'Business', 'Sci/Tech']

print(next(train_dataset))