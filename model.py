import torch
import torch.nn as nn
import torch.nn.functional as F

n_classes = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
        self.fc = nn.Linear(1000, n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return self.softmax(x)
