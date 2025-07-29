# -*- encoding: utf-8 -*-

import os
import sys
import argparse
import glob
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from spikingjelly.activation_based import functional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SNN

torch.manual_seed(2024)

parser = argparse.ArgumentParser(description='Init with SNN')
parser.add_argument('--train_data', type=str, default='./datasets/train_data',
                    help='Path to the training data directory')
parser.add_argument('--controls', type=str, nargs='+', default=['standard', 'diff_shape', 'same_area'],
                    help='List of control conditions to include in the dataset')
parser.add_argument('--max_number', type=int, default=30,
                    help='Maximum number to include in the dataset')
parser.add_argument('--batch_size', type=int, default=30,
                    help='Batch size for training and testing')
parser.add_argument('--epochs', type=int, default=1,
                    help='Number of epochs to train the model')
parser.add_argument('--num_workers', type=int, default=2,
                    help='Number of worker threads for data loading')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to run the model on (e.g., cuda:0, cpu)')
parser.add_argument('--T', type=int, default=16,
                    help='Number of time steps for the SNN')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='Learning rate for the gradient descent optimizer')
parser.add_argument('--save_dir', type=str, default='./model_set',
                    help='Directory to save the trained model checkpoints')
args = parser.parse_args()

class NumberSense(Dataset):
    """
    A custom PyTorch Dataset for loading and preprocessing images representing numbers 
    for number sense tasks.
    Args:
        _path (str): Root directory containing image folders.
        _controls (list): List of folder names to include as controls.
        max_number (int): Maximum number label to stimulus.
        transforms (callable, optional): Optional transform to be applied on an 
        image (e.g., torchvision transforms).
        img_format (str, optional): Image file format (default: 'jpg').
    Attributes:
        data (torch.Tensor): Tensor containing image data, normalized and converted 
        to grayscale or transformed.
        labels (torch.Tensor): Tensor containing integer labels extracted from image filenames.
    Methods:
        __getitem__(idx): Returns the image and label at the specified index.
        __len__(): Returns the total number of samples in the dataset.
    """
    def __init__(self, _path, _controls, max_number, transforms=None, img_format='jpg'):
        img_format = '*.' + img_format
        names = []
        for c in _controls:
            names.extend(glob.glob(os.path.join(_path, c, img_format)))
        # names.sort()
        names = [name for name in names if int(os.path.basename(name).split('.')[0].split('_')[0]) <= max_number]
        # the labels are 0, 1, 2 for the three controls
        self.labels = torch.Tensor([0 if _controls[0] in name else 1 if _controls[1] in name else 2 for name in names]).to(torch.int64)
        if transforms:
            data = np.array([np.array(transforms(Image.open(name).convert('RGB'))) for name in names])
        else:
            data = np.array([np.array(Image.open(name).convert('L')) for name in names])
        self.data = torch.Tensor(data)
        self.data = 255. - self.data
        # convert 'L'
        self.data = self.data.unsqueeze(1)
        # self.data = self.data.permute(0, 1, 2, 3)
        # # convert 'RGB'
        # self.data = self.data.permute(0, 3, 1, 2)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.data)


def main():

    # model initialization
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = SNN(T=args.T, num_classes=len(args.controls)).to(device=device)
    
    
    # training 1 epoch
    epochs = args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    dataset = NumberSense(args.train_data, args.controls, args.max_number)
    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(trainloader):
            optimizer.zero_grad()
            labels = y.to(device=device)
            x = x.to(device=device)
            predict = model(x)
            loss = F.cross_entropy(predict, labels)
            loss.backward()
            optimizer.step()
            functional.reset_net(model)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # save the model
    save_path = os.path.join(args.save_dir, 'init_SNN_t{}_model.pth'.format(args.T))
    checkpoint = {
        'model': model.state_dict(),
    }
    torch.save(checkpoint, save_path)

if __name__ == '__main__':
    main()
