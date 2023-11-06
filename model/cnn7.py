# model for CIFAR 10

import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 3)
        self.conv2 = nn.Conv2d(96, 96, 3, stride=2)
        self.conv3 = nn.Conv2d(96, 192, 1)
        self.conv4 = nn.Conv2d(192, 10, 1)
        self.fc1 = nn.Linear(1960, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 2)
        self.af = F.relu

    def forward(self, x):
        h = self.conv1(x)
        h = self.af(h)
        h = self.conv2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.af(h)
        h = h.view(-1, 1960)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        return self.fc3(h)


class CNN_STL(nn.Module):
    def __init__(self):
        super(CNN_STL, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(6, 16, 5)
        self.conv4 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        self.sigmoid = nn.Sigmoid()
        self.m = nn.Dropout2d(0.2)
        self.n = nn.Dropout(0.2)
        self.b1 = nn.BatchNorm2d(6)
        self.b2 = nn.BatchNorm2d(16)
        self.b3 = nn.BatchNorm1d(120)
        self.b4 = nn.BatchNorm1d(84)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def cnn_cifar():
    return CNN_CIFAR()


def cnn_stl():
    return CNN_STL()

