# -*- encoding: utf-8 -*-
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from spikingjelly.clock_driven import functional

from anova import FastAnovaRunner
from model import SNN
from utils import get_preferred_numerosity, rns_with_average_rankings

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

torch.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser(description='Precise Learning with SNN')
parser.add_argument('--data_path', type=str, default='./datasets/zipf_data',
                    help='Path to the data directory')
parser.add_argument('--controls', type=str, nargs='+', default=['standard', 'diff_shape', 'same_area'],
                    help='List of control conditions to include in the dataset')
parser.add_argument('--max_number', type=int, default=30,
                    help='Maximum number numerosity to include in the dataset')
parser.add_argument('--num_workers', type=int, default=2,
                    help='Number of worker threads for data loading')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to run the model on (e.g., cuda:0, cpu)')
parser.add_argument('--T', type=int, default=16,
                    help='Number of time steps for the SNN')
parser.add_argument('--batch_size', type=int, default=20,
                    help='Batch size for training and testing')
parser.add_argument('--model_path', type=str, default='./model_set/init_SNN_t16_model.pth',
                    help='Path to the initial model checkpoint')
parser.add_argument('--save_result', action='store_true', default=True,
                    help='Whether to save the result of the analysis')
parser.add_argument('--save_dir', type=str, default='./analyze_data',
                    help='Directory to save the analysis results')
parser.add_argument('--hook_layers', type=str, nargs='+', default=['spike_features.0.3', 'spike_features.1.3', 'spike_features.2.3', 'spike_features.3.3', 'spike_features.4.3'],
                    help='List of layer names to hook for feature extraction')
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
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.data)

features_out_hook = []

def hook(module, fea_in, fea_out):
    """
    A forward hook function to capture and store the output features of a module during a forward pass.

    Args:
        module (torch.nn.Module): The module to which the hook is attached.
        fea_in (tuple): The input to the module.
        fea_out (torch.Tensor or tuple): The output from the module.

    Returns:
        None

    Side Effects:
        Appends the output features (fea_out) to the global list `features_out_hook`.
    """

    features_out_hook.append(fea_out) # output of the layer
    return None

def main():
    print('timestep T:', args.T)

    # model initialization
    model = SNN(T=args.T, num_classes=args.max_number)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model_state_dict = model.state_dict()
    load_state_dict = {k: v for k, v in checkpoint['model'].items() if k in model_state_dict and 'spike_classifier.7' not in k}
    model_state_dict.update(load_state_dict)
    model.load_state_dict(model_state_dict)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # use hook to get layer output, you can use print(model) to get layer name which you want to see
    for (name, module) in model.named_modules():
        if name in args.hook_layers:
            module.register_forward_hook(hook=hook)
    
    all_numerositys = []
    all_controls = []
    all_output = []

    model.eval()
    for control_id, control in enumerate(args.controls):
        dataset = NumberSense(args.data_path, [control], args.max_number)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                x = x.to(device)
                labels = y.detach().numpy()
                predict = model(x)
                # hook size: Time_step * batch_size * channel * H * W
                all_neurons = features_out_hook[0].sum(0).detach().cpu().numpy().reshape(len(y), -1)
                for i in range(1, len(features_out_hook)):
                    all_neurons = np.hstack((all_neurons, features_out_hook[i].sum(0).detach().cpu().numpy().reshape(len(y), -1)))
                
                all_numerositys.append(labels)
                all_controls.append(np.full(len(labels), control_id))
                all_output.append(all_neurons)
                features_out_hook.clear()
                functional.reset_net(model)

    # concatenate all results
    all_numerositys = np.concatenate(all_numerositys, axis=0)
    all_controls = np.concatenate(all_controls, axis=0)
    all_output = np.concatenate(all_output, axis=0)

    # run ANOVA analysis
    anova_runner = FastAnovaRunner(p1=0.01, p2=0.01, p3=0.01, numerositys=all_numerositys, controls=all_controls, responses=all_output)
    anova_runner.run()
    neuron_number = all_output.shape[1]
    print('NSN neuron number: {}, NSN radio: {}'.format(len(anova_runner.PN_index), len(anova_runner.PN_index) / neuron_number))

    # get preferred numerosity and selectivity
    all_response = np.transpose(all_output, [1, 0])
    _, origin_responses, PNs = get_preferred_numerosity(all_response, all_numerositys, all_controls, anova_runner.PN_index)
    selectivity = []
    for response, pn in zip(origin_responses, PNs):
        rns_numerosity = rns_with_average_rankings(response, all_numerositys)
        selectivity.append(rns_numerosity[pn - 1])
    print('selectivity: ', np.array(selectivity).mean())

    if args.save_result:
        # save results
        model_name = os.path.basename(args.model_path).replace('.pth', '')
        save_path = os.path.join(args.save_dir, 'NSN_results_{}.npz'.format(model_name))
        print('Saving results to:', save_path)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        np.savez(save_path, all_labels=all_numerositys, all_controls=all_controls,
                 all_output=all_output, PN_index = np.array(anova_runner.PN_index))


if __name__ == '__main__':
    main()
