# -*- encoding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_preferred_numerosity, rns_with_average_rankings

parser = argparse.ArgumentParser(description='Numerical Distance Effect Analysis')
parser.add_argument('--data_path', type=str, default='./analyze_data/NSN_results_init_SNN_t16_model.npz',
                    help='Path to the data file containing analysis results')
args = parser.parse_args()

def numerical_distance_effect():
    """
    Analyzes the numerical distance effect by computing the average response of preferred numerosity neurons
    across different numerical distances.
    Loads data from a specified file path, extracts relevant arrays, and calculates the response of neurons
    with preferred numerosity to various numerical distances. The function aggregates and averages the
    responses (RNS: response normalized scores) for each distance, then prints the results.
    """

    save_data = np.load(args.data_path)
    PN_index = save_data['PN_index']
    all_numerositys = save_data['all_labels']
    all_controls = save_data['all_controls']
    all_output = save_data['all_output']
    all_response = np.transpose(all_output, [1, 0])
    _, origin_responses, PN_number = get_preferred_numerosity(all_response, all_numerositys, all_controls, PN_index)
    unique_numerosity = np.unique(all_numerositys).astype(np.int32)
    
    distances_rns = np.zeros(59)
    count_rns = np.zeros(59)
    for numerosity in unique_numerosity:
        responses = origin_responses[PN_number == numerosity]
        calculate_index = PN_index[PN_number == numerosity]
        for index, response in tqdm(zip(calculate_index, responses)):
            rns = rns_with_average_rankings(response, all_numerositys)
            distances_rns[30-numerosity:60-numerosity] += rns
            count_rns[30-numerosity:60-numerosity] += 1

    for i in range(1, 30):
        distances_rns[i + 29] += distances_rns[29 - i]
        count_rns[i + 29] += count_rns[29 - i]
    distances_rns = distances_rns[29:]
    count_rns = count_rns[29:]
    distances_rns /= count_rns

    for index, rns in zip(range(30), distances_rns):
        print("distance: {}, rns: {}".format(index, rns))


def main():
    numerical_distance_effect()

if __name__ == '__main__':
    main()



