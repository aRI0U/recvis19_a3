# imports
import argparse
from datetime import datetime
import glob
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

# Data initialization and loading
from data import data_transforms, StudentDataset
from log import Log
from model import Net

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default=None, metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--save_freq', type=int, default=10, metavar='f',
                    help='frequency the model weights are saved')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)

# Create experiment folder
if args.experiment is None:
    args.experiment = 'exp_%04d' % (len(glob.glob('exp_*'))+1)
os.makedirs(args.experiment, exist_ok=True)
log = Log(args.experiment)

train_set = datasets.ImageFolder(
    os.path.join(args.data, 'train_images'),
    transform=data_transforms['train']
)

val_set = datasets.ImageFolder(
    os.path.join(args.data, 'val_images'),
    transform=data_transforms['test']
)

train_loader = torch.utils.data.DataLoader(train_set,
    batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set,
    batch_size=1, shuffle=False, num_workers=4)

# Neural network and optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
model = Net()
if use_cuda:
    print('Using GPU')
    model.cuda()
else:
    print('Using CPU')

optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.1,
    patience=10,
    verbose=True,
    threshold=0.01,
    threshold_mode='abs'
)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        validation_loss += criterion(output, target).data.item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    log.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return correct/len(val_loader.dataset)


for epoch in range(1, args.epochs + 1):
    if epoch%20 == 0:
        student_set = StudentDataset(
            os.path.join(args.data, 'test_images'),
            model,
            model.state_dict(),
            ratio=epoch/200,
            transform=data_transforms
        )
        concat_set = torch.utils.data.ConcatDataset([train_set, student_set])
        train_loader = torch.utils.data.DataLoader(concat_set,
            batch_size=args.batch_size, shuffle=True, num_workers=1)

    train(epoch)
    val_score = validation()
    scheduler.step(val_score)
    if epoch % args.save_freq == 0:
        model_file = os.path.join(args.experiment, 'model_%d.pth' % epoch)
        torch.save(model.state_dict(), model_file)
