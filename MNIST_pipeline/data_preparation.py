import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.getcwd())
import package as pkg
from benchmarking import bm


@pkg.MeasureTime(bm, description="time spent on preparing data")
@pkg.MeasureMemorySamples(bm, description="memory usage of data preparation")
## @pkg.MeasureMemoryTracemalloc(bm, description="memory usage of data preparation")
def data_preparation():

    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 30

    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # choose the training and test datasets
    trainset = datasets.MNIST(root='MNIST data', train=True,
                                       download=True, transform=transform)
    testset = datasets.MNIST(root='MNIST data', train=False,
                                      download=True, transform=transform)

    # prepare data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        num_workers=num_workers)


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # linear layer (784 -> 1 hidden node)
            self.fc1 = nn.Linear(28 * 28, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, 10)
            self.dropout = nn.Dropout(p=.5)

        def forward(self, x):
            # flatten image input
            x = x.view(-1, 28 * 28)
            # add hidden layer, with relu activation function
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = F.log_softmax(self.fc3(x), dim=1)

            return x

    # initialize the NN
    model = Net()

    return [model, trainloader, testloader]




