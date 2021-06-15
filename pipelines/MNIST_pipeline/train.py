import torch
import torch.nn as nn
import os
import sys
import umlaut
from benchmarking import bm

@umlaut.MeasureTime(bm, description="Training time")
@umlaut.MeasureThroughput(bm, description="Training throughput")
@umlaut.MeasureLatency(bm, description="Training latency")
@umlaut.MeasureMemorySamples(bm, description="Training memory usage")
# @pkg.MeasureMemoryTracemalloc(bm, description="Training memory usage")
def train(model, trainloader):

    n_epochs = 10  # suggest training between 20-50 epochs

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    model.train()  # prep model for training

    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for data, target in trainloader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item() * data.size(0)

        # print training statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(trainloader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch + 1,
            train_loss
        ))

    return {"model": model, "num_entries": len(trainloader)}
