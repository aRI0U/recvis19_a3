import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

data_transforms = {}

data_transforms['train'] = transforms.Compose([
    transforms.Resize(320),
    transforms.RandomResizedCrop(288),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
    transforms.RandomErasing()
])

data_transforms['test'] = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(288),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
