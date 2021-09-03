import torch
import torchtext
import numpy as np
from torchnlp import *
train_dataset, test_dataset, classes, vocab = load_dataset()
vocab_size = len(vocab)
print("Vocab size = ",vocab_size)