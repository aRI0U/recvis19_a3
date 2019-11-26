import glob
import numpy as np
import os
import PIL.Image as Image

import torch
import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 288 x 288 in size
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

class StudentDataset(torch.utils.data.Dataset):
    def __init__(self, root, model, state_dict=None, threshold=0.5, transform=None):
        super(StudentDataset, self).__init__()
        self.root = root
        self.transform = transform
        if state_dict:
            model.load_state_dict(state_dict)
        model.eval()
        self.images = []
        self.classes = []
        scores = []

        # determine for each unlabeled image the predicted class and its associated uncertainty
        for file in glob.iglob(self.root + '/*/*.jpg'):
            image = Image.open(file)
            self.images.append(image)
            if self.transform:
                image = self.transform['test'](image)
                image = image.view((1,image.size(0),image.size(1),image.size(2)))
            if torch.cuda.is_available():
                image = image.cuda()
            distribution = model(image).detach().cpu().numpy().reshape(-1)
            score, cat = np.amax(distribution), np.argmax(distribution)
            distribution[cat] = 0
            score2 = distribution.max()
            self.classes.append(cat)
            scores.append(score2/score)

        self.classes = np.array(self.classes)
        scores = np.array(scores)

        # keep only the less uncertain results
        last_idx = len(scores[scores<threshold])
        self.partition = np.argpartition(scores, last_idx)
        scores = scores[self.partition]
        self.classes = self.classes[self.partition][:last_idx]
        self.mu_max = scores[last_idx-1]

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.images[self.partition[idx]]
        target = self.classes[idx]
        if self.transform:
            sample = self.transform['train'](sample)

        return sample, target
