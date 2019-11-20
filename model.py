import torch
import torch.nn as nn
import torch.nn.functional as F

n_classes = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        with torch.no_grad():
            x = self.resnet(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)
