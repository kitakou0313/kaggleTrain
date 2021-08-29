import torch
import torchtext
import os
import collections
os.makedirs('./data',exist_ok=True)
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data', split=('train', 'test'))
classes = ['World', 'Sports', 'Business', 'Sci/Tech']

train_dataset = list(train_dataset)
test_dataset = list(test_dataset)

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

counter = collections.Counter()
for (lavel, line) in train_dataset:
    counter.update(tokenizer(line))

vocab = torchtext.vocab.vocab(counter, min_freq=1)

vocab_size = len(vocab)
print(f"Vocab size if {vocab_size}")

def encode(x):
    return [vocab.get_stoi()[s] for s in tokenizer(x)]

print(encode('I love to play with my words'))

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
corpus = [
    'I like hot dogs.',
    'The dog ran fast.',
    'Its hot outside.',
]
vectorizer.fit_transform(corpus)
print(vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray())

def to_bow(text,bow_vocab_size=vocab_size):
    res = torch.zeros(bow_vocab_size,dtype=torch.float32)
    for i in encode(text):
        if i<bow_vocab_size:
            res[i] += 1
    return res

from torch.utils.data import DataLoader
import numpy as np 

def bowify(b):
    return (
        torch.LongTensor([t[0]-1 for t in b]),
        torch.stack([to_bow(t[1]) for t in b])
    )

train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=bowify, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=bowify, shuffle=True)

