import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, window_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(window_size, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(24320, 100)      # (32*1*379, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 10)
        self.bn4 = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(p=0.1) #########
        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        #x = self.pool1(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.flat(x)         #x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.dropout(x) #########
        x = self.bn3(x)
        x = self.fc2(x)
        x = self.dropout(x) #########
        x = self.bn4(x)
        
        x = self.output(x)
        x = self.sigmoid(x)
        return x.squeeze()
