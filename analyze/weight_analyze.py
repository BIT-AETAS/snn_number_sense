# -*- encoding: utf-8 -*-
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_PN_connection, get_PN_connection_with_pooling, get_preferred_numerosity

parser = argparse.ArgumentParser(description='Analyze the weight of the model')
parser.add_argument('--analyze_data_path', type=str, default='./analyze_data/NSN_results_init_SNN_t4_model.npz',
                    help='Path to the analyze data file')
parser.add_argument('--model_path', type=str, default='./model_set/init_SNN_t4_model.pth',
                    help='Path to the model file')
args = parser.parse_args()

def main():    
    # Load the model and analyze data
    model = torch.load(args.model_path, map_location='cpu')
    save_data = np.load(args.analyze_data_path)
    PN_index = save_data['PN_index']
    PN_index = np.array(PN_index).astype(int)
    all_labels = save_data['all_labels']
    all_controls = save_data['all_controls']
    all_output = save_data['all_output']
    all_response = np.transpose(all_output, [1, 0])
    avg_responses, origin_responses, PN_number = get_preferred_numerosity(all_response, all_labels, all_controls, PN_index)
    
    PN_map = {PN_index[i]: PN_number[i] for i in range(len(PN_index))}

    # layer_size = [64*56*56, 192*28*28, 384*14*14, 256*14*14, 256*14*14]
    layer_size = [200704, 150528, 75264, 50176, 50176]
    # layer_index = [0, 200704, 200704+150528, 200704+150528+75264, 200704+150528+75264+50176, 200704+150528+75264+50176+50176]
    layer_index = [0, 200704, 351232, 426496, 476672, 526848]
    layer_name = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

    # Get the number of PN units in each layer
    layer_PN_unit = []
    for i in range(5):
        count_PN_unit = PN_index[PN_index < layer_index[i+1]]
        count_PN_unit = count_PN_unit[count_PN_unit >= layer_index[i]]
        layer_PN_unit.append(count_PN_unit)

    model_weight = model['model']
    layer_PN_connections = []

    # kernel info 1 to 2
    kernel_info_1_to_2 = {'k_h': 5, 'k_w': 5, 's_h': 1, 's_w': 1, 'p_h': 2, 'p_w': 2, 'd_h': 1, 'd_w': 1,
                            'in_channel': 64, 'out_channel': 192, 'output_h': 28, 'output_w': 28, 'input_h': 28, 'input_w': 28,
                            'pool_kernel_size': 2, 'pool_stride': 2, 'input_pool_h': 56, 'input_pool_w': 56}
    layer2_PN_connections = get_PN_connection_with_pooling(layer_PN_unit[1], layer_index[1], layer_index[0], model_weight['spike_features.1.1.weight'], kernel_info_1_to_2)
    layer_PN_connections.extend(layer2_PN_connections)
    # kernel info 2 to 3
    kernel_info_2_to_3 = {'k_h': 3, 'k_w': 3, 's_h': 1, 's_w': 1, 'p_h': 1, 'p_w': 1, 'd_h': 1, 'd_w': 1,
                            'in_channel': 192, 'out_channel': 384, 'output_h': 14, 'output_w': 14, 'input_h': 14, 'input_w': 14,
                            'pool_kernel_size': 2, 'pool_stride': 2, 'input_pool_h': 28, 'input_pool_w': 28}
    layer3_PN_connections = get_PN_connection_with_pooling(layer_PN_unit[2], layer_index[2], layer_index[1], model_weight['spike_features.2.1.weight'], kernel_info_2_to_3)
    layer_PN_connections.extend(layer3_PN_connections)
    # kernel info 3 to 4
    kernel_info_3_to_4 = {'k_h': 3, 'k_w': 3, 's_h': 1, 's_w': 1, 'p_h': 1, 'p_w': 1, 'd_h': 1, 'd_w': 1,
                        'in_channel': 384, 'out_channel': 256, 'output_h': 14, 'output_w': 14, 'input_h': 14, 'input_w': 14}
    layer4_PN_connections = get_PN_connection(layer_PN_unit[3], layer_index[3], layer_index[2], model_weight['spike_features.3.1.weight'], kernel_info_3_to_4)
    layer_PN_connections.extend(layer4_PN_connections)
    # kernel info 4 to 5
    kernel_info_4_to_5 = {'k_h': 3, 'k_w': 3, 's_h': 1, 's_w': 1, 'p_h': 1, 'p_w': 1, 'd_h': 1, 'd_w': 1,
                        'in_channel': 256, 'out_channel': 256, 'output_h': 14, 'output_w': 14, 'input_h': 14, 'input_w': 14}
    layer2_PN_connections = get_PN_connection(layer_PN_unit[4], layer_index[4], layer_index[3], model_weight['spike_features.4.1.weight'], kernel_info_4_to_5)
    layer_PN_connections.extend(layer2_PN_connections)

    layer_PN_unit_set = set(PN_index)
    PN_inner_weights = []
    PN_PN_weights = []
    PN_other_weights = []
    PN_weights = []
    for unit in tqdm(layer_PN_connections):
        for connection in unit:
            if connection[1] in layer_PN_unit_set:
                if PN_map[connection[0]] == PN_map[connection[1]]:
                    PN_inner_weights.append(connection[2])
                else:
                    PN_PN_weights.append(connection[2])
            else:
                PN_other_weights.append(connection[2])
            PN_weights.append(connection[2])
    # Calculate mean of all weights in the model
    all_weight_mean = torch.cat([
        model_weight['spike_features.1.1.weight'].view(-1),
        model_weight['spike_features.2.1.weight'].view(-1),
        model_weight['spike_features.3.1.weight'].view(-1),
        model_weight['spike_features.4.1.weight'].view(-1)
    ]).mean()

    print(f"All weight mean: {all_weight_mean.item():.6f}")
    print(f"PN inner weights mean: {np.mean(PN_inner_weights):.6f}")
    print(f"PN-PN weights mean: {np.mean(PN_PN_weights):.6f}")
    print(f"PN-other weights mean: {np.mean(PN_other_weights):.6f}")
    print(f"All PN weights mean: {np.mean(PN_weights):.6f}")


if __name__ == '__main__':
    main()