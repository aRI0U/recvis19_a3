import torch
import torch.nn as nn

import torchvision

n_classes = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = torchvision.models.resnext101_32x8d(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x
