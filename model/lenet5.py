# model for FashionMNIST
import torch
import torch.nn as nn

class LeNet_FMNIST(nn.Module):
    def __init__(self):
        super(LeNet_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # in_size = x.size(0)
        # out = self.relu(self.mp(self.conv1(x)))
        # out = self.relu(self.mp(self.conv2(out)))
        # out = self.relu(self.conv3(out))
        # out = out.view(in_size, -1)
        # out = self.relu(self.bn_fc1(self.fc1(out)))
        # return self.fc2(out)
        out = self.conv1(x)
        out = self.mp(out)
        out = self.conv2(out)
        out = self.mp(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.bn_conv1 = nn.BatchNorm2d(6)
        self.bn_conv2 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.bn_fc1 = nn.BatchNorm1d(84)

        self.layer1 = nn.Sequential(self.conv1, self.mp, self.relu)
        self.layer2 = nn.Sequential(self.conv2, self.mp, self.relu)
        self.layer3 = nn.Sequential(self.conv3, self.relu)
        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3])
        self.layer4 = nn.Sequential(self.fc1, self.bn_fc1, self.relu)
        self.classifier = nn.Linear(84, 2)

    def forward(self, x):
        h = x
        for i, layer_module in enumerate(self.layers):
            h = layer_module(h)
        h = h.view(h.size(0), -1)
        h = self.layer4(h)
        h = self.classifier(h)
        return h


def lenet_fmnist():
    return LeNet()