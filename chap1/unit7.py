import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

traningData = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

testData = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

trainDataLoader = DataLoader(traningData, batch_size=64)
testDataLoader = DataLoader(testData, batch_size=64)

class NN(nn.Module):
    """
    自作NN
    """
    def __init__(self):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()

        self.linearReluStack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        """
        順方向
        """
        x = self.flatten(x)
        logits = self.linearReluStack(x)
        return logits


model = NN()

learningRate = 1e-3
batchSize = 64
epoch = 5

lossFn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

def trainLoop(dataLoder:DataLoader, model:NN, lossFn:nn.CrossEntropyLoss, optimizer:torch.optim.SGD):
    """
    学習ループ
    """
    size = len(dataLoder.dataset)
    for batch, (X, y) in enumerate(dataLoder):
        pred = model(X)
        loss = lossFn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def testLoop(dataLoader:DataLoader, model:NN, lossFn:nn.CrossEntropyLoss):
    """
    テスト用ループ
    """
    size = len(dataLoader.dataset)
    testLoss, correct = 0,0

    with torch.no_grad():
        for X, y in dataLoader:
            pred = model(X)
            testLoss += lossFn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    testLoss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {testLoss:>8f} \n")
        

epochs = 10
for t in range(epochs):
    print("Epoch;", t)

    trainLoop(trainDataLoader, model, lossFn, optimizer)
    testLoop(testDataLoader, model, lossFn)

print("Done!")