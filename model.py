import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

n_classes = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = torchvision.models.resnext101_32x8d(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
