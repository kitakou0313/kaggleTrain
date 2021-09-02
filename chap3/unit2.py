import torch
from torch import optim
import torchtext
from tqdm import tqdm
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

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = CountVectorizer()
corpus = [
    'I like hot dogs.',
    'The dog ran fast.',
    'Its hot outside.',
]
#vectorizer.fit_transform(corpus)
#print(vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray())

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

net = torch.nn.Sequential(
    torch.nn.Linear(vocab_size, 4),
    torch.nn.LogSoftmax(dim=1)
)

def train_epoch(net:torch.nn.Sequential, datalodaer:DataLoader, lr=0.01,optimizer=None,loss_fn:torch.nn.NLLLoss = torch.nn.NLLLoss(),epoch_size=None, report_freq=200):
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    net.train()
    totalLoss, acc, count, i = 0,0,0,0

    bar = tqdm(total = epoch_size)

    for label, feature in datalodaer:
        optimizer.zero_grad()
        out = net(feature)
        loss = loss_fn(out, label)
        loss.backward()
        optimizer.step()
        totalLoss += loss
        _,predicted = torch.max(out,1)
        acc+=(predicted==label).sum()
        count+=len(label)
        i+=1
        bar.update(1)

        if i%report_freq==0:
            print(f"{count}: acc={acc.item()/count}")
        if epoch_size and count>epoch_size:
            break
    return totalLoss.item()/count, acc.item()/count


# train_epoch(net, train_loader, epoch_size=1500)

bigram_vectorizer = CountVectorizer(ngram_range=(1,2), token_pattern=r'\b\w+\b', min_df=1)
corpus = [
        'I like hot dogs.',
        'The dog ran fast.',
        'Its hot outside.',
    ]
# bigram_vectorizer.fit_transform(corpus)
# print(bigram_vectorizer.vocabulary_)
# bigram_vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()

counter = collections.Counter()
for (label, line) in train_dataset:
    l = tokenizer(line)
    counter.update(torchtext.data.utils.ngrams_iterator(l, ngrams=2))
bi_vocab = torchtext.vocab.vocab(counter,min_freq=1)

print("Bican length, num:",len(bi_vocab))

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
vectorizer.fit_transform(corpus)
print(vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray())