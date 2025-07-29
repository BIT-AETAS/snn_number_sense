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
from spikingjelly.activation_based import functional, layer, learning

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import SNN

torch.manual_seed(2024)

parser = argparse.ArgumentParser(description='Approximate Learning with SNN')
parser.add_argument('--train_data', type=str, default='./datasets/train_data',
                    help='Path to the training data directory')
parser.add_argument('--test_data', type=str, default='./datasets/test_data',
                    help='Path to the testing data directory')
parser.add_argument('--controls', type=str, nargs='+', default=['standard', 'diff_shape', 'same_area'],
                    help='List of control conditions to include in the dataset')
parser.add_argument('--max_number', type=int, default=30,
                    help='Maximum number label to include in the dataset')
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size for training and testing')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train the model')
parser.add_argument('--num_workers', type=int, default=2,
                    help='Number of worker threads for data loading')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to run the model on (e.g., cuda:0, cpu)')
parser.add_argument('--T', type=int, default=16,
                    help='Number of time steps for the SNN')
parser.add_argument('--tau_pre', type=float, default=2.0,
                    help='Time constant for pre-synaptic spike decay')
parser.add_argument('--tau_post', type=float, default=100.0,
                    help='Time constant for post-synaptic spike decay')
parser.add_argument('--model_path', type=str, default='./model_set/init_SNN_t16_model.pth',
                    help='Path to the initial model checkpoint')
parser.add_argument('--stdp_lr', type=float, default=0.00001,
                    help='Learning rate for the STDP optimizer')
parser.add_argument('--lr', type=float, default=0.001,
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
        self.labels = torch.Tensor([int(os.path.basename(name).split('.')[0].split('_')[0]) for name in names]).to(torch.int64)
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
    
def f_weight(x):
    """
    Clamps the input tensor values to the range [-1, 1].

    Args:
        x (torch.Tensor): Input tensor to be clamped.

    Returns:
        torch.Tensor: Tensor with values clamped between -1 and 1.
    """
    return torch.clamp(x, -1, 1.)

def main():
    # load the dataset
    dataset = NumberSense(args.train_data, args.controls, args.max_number)
    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testdataset = NumberSense(args.test_data, args.controls, args.max_number)
    testloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # model initialization
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = SNN(T=args.T, num_classes=args.max_number)

    # load the initial model checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model_state_dict = model.state_dict()
    load_state_dict = {k: v for k, v in checkpoint['model'].items() if k in model_state_dict and 'spike_classifier.7' not in k}
    model_state_dict.update(load_state_dict)
    model.load_state_dict(model_state_dict)

    model = model.to(device=device)
    
    epochs = args.epochs

    # Initialize the STDP learners
    instances_stdp = (layer.Conv2d, layer.Linear)
    tau_pre = args.tau_pre
    tau_post = args.tau_post
    stdp_learners = []
    step_mode = 'm'
    for i in range(model.spike_features.__len__()):
        if isinstance(model.spike_features[i], layer.MaxPool2d):
            continue
        for j in range(model.spike_features[i].__len__()):
            if isinstance(model.spike_features[i][j], instances_stdp):
                stdp_learners.append(
                    learning.STDPLearner(step_mode=step_mode, synapse=model.spike_features[i][j], sn=model.spike_features[i][j+2], tau_pre=tau_pre, tau_post=tau_post,
                                        f_pre=f_weight, f_post=f_weight)
                )
    for i in range(model.spike_classifier.__len__() - 1):
        if isinstance(model.spike_classifier[i], instances_stdp):
            stdp_learners.append(
                learning.STDPLearner(step_mode=step_mode, synapse=model.spike_classifier[i], sn=model.spike_classifier[i+1], tau_pre=tau_pre, tau_post=tau_post,
                                    f_pre=f_weight, f_post=f_weight)
            )
    params_stdp = []
    for m in model.spike_features.modules():
        if isinstance(m, instances_stdp):
            for p in m.parameters():
                params_stdp.append(p)
    for m in model.spike_classifier.modules():
        if isinstance(m, instances_stdp):
            for p in m.parameters():
                params_stdp.append(p)

    params_stdp_set = set(params_stdp)
    params_gradient_descent = []
    for p in model.parameters():
        if p not in params_stdp_set:
            params_gradient_descent.append(p)
    optimizer_stdp = torch.optim.SGD(params_stdp, lr=args.stdp_lr, momentum=0.)

    # initialize the optimizer for gradient descent
    optimizer = torch.optim.Adam(params_gradient_descent, lr=args.lr)

    # save directory
    save_dir = os.path.join(args.save_dir, 'stdp_lr{}'.format(args.stdp_lr))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # training loop
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for i in range(stdp_learners.__len__()):
            stdp_learners[i].reset()

        for x, y in tqdm(trainloader):
            optimizer.zero_grad()
            optimizer_stdp.zero_grad()
            labels = y.to(device=device)
            x = x.to(device=device)
            predict = model(x)
            loss = F.cross_entropy(predict, labels - 1)
            loss.backward()

            optimizer_stdp.zero_grad()
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].step(on_grad=True)
            optimizer.step()
            optimizer_stdp.step()

            # reset netword and stdp learner
            functional.reset_net(model)
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].reset()
        
        optimizer.zero_grad()
        optimizer_stdp.zero_grad()

        # evaluate the model
        model.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for x, y in tqdm(testloader):
                labels = y.to(device=device)
                x = x.to(device=device)
                predict = model(x)
                loss = F.cross_entropy(predict, labels - 1)

                test_samples += labels.numel()
                test_loss += loss.item() * labels.numel()
                test_acc += (predict.argmax(1) + 1 == labels).float().sum().item()
                functional.reset_net(model)
                for i in range(stdp_learners.__len__()):
                    stdp_learners[i].reset()
                optimizer_stdp.zero_grad()

        test_loss /= test_samples
        test_acc /= test_samples
        
        # save the model checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        save_path = os.path.join(save_dir, 'SNN_t{}_model_stdp_epoch{}.pth'.format(args.T, epoch))
        torch.save(checkpoint, save_path)

if __name__ == '__main__':
    main()
