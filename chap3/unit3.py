import torch
import torchtext
import numpy as np
from torchnlp import *
train_dataset, test_dataset, classes, vocab = load_dataset()
vocab_size = len(vocab)
print("Vocab size = ",vocab_size)

class EmbedClassifier(torch.nn.Module):
    """
    embeddig層を追加
    """
    def __init__(self, vocab_size:int, embed_dim:int, num_class:int):
        """
        docstring
        """
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc = torch.nn.Linear(embed_dim, num_class)

    def forward(self, x):
        """
        docstring
        """
        x = self.embedding(x)
        x = torch.mean(x, dim=1)

        return self.fc(x)