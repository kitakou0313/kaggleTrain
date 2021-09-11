import torch
import torchtext
import numpy as np
import torch.utils.data
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=padify, shuffle=True)

net = EmbedClassifier(vocab_size, 32, len(classes)).to(device)
train_epoch(net, train_loader, lr=1, epoch_size=25000)

class EmbedClassifier(torch.nn.Module):
    """
    docstring
    """
    def __init__(self, vocab_size, embed_dim, num_class):
        """
        docstring
        """
        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(
            vocab_size, embed_dim
        )
        self.fc = torch.nn.Linear(embed_dim, num_class)

    def forward(self, text, off):
        """
        docstring
        """
        x = self.embedding(text, off)
        return self.fc(x)
