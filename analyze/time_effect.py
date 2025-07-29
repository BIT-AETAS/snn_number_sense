# -*- encoding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_preferred_numerosity, rns_with_average_rankings

parser = argparse.ArgumentParser(description='Numerical Distance Effect Analysis')
parser.add_argument('--data_paths', type=str, nargs='+', default=['./analyze_data/NSN_results_init_SNN_t4_model.npz',
                                                                  './analyze_data/NSN_results_init_SNN_t8_model.npz',
                                                                  './analyze_data/NSN_results_init_SNN_t16_model.npz',
                                                                  './analyze_data/NSN_results_init_SNN_t32_model.npz'],
                    help='List of data file paths to analyze')
parser.add_argument('--timesteps', type=int, nargs='+', default=[4, 8, 16, 32],
                    help='List of timesteps corresponding to data')
args = parser.parse_args()

def numerical_time_effect():
    """
    Analyzes the effect of different time steps on numerical selectivity.
    This function iterates over provided data paths and corresponding time steps, loading neural response data for each.
    For each dataset, it computes the selectivity of neural responses to different numerosities using a ranking-based
    selectivity metric. The selectivity values are aggregated and associated with the logarithm (base 2) of the time step.
    Finally, the function prints the mean selectivity grouped by time step.
    Assumes the existence of the following external variables and functions:
        - args.data_paths: List of file paths to .npz data files.
        - args.timesteps: List of time step values corresponding to each data file.
        - get_preferred_numerosity: Function to compute preferred numerosity and responses.
        - rns_with_average_rankings: Function to compute selectivity ranking.
        - np: NumPy module.
        - pd: pandas module.
    Returns:
        None
    """
    selectivity = []
    timestep_data = []
    for data_path, timestep in zip(args.data_paths, args.timesteps):
        save_data = np.load(data_path)
        PN_index = save_data['PN_index']
        all_labels = save_data['all_labels']
        all_controls = save_data['all_controls']
        all_output = save_data['all_output']
        all_response = np.transpose(all_output, [1, 0])
        _, origin_responses, PN_number = get_preferred_numerosity(all_response, all_labels, all_controls, PN_index)

        for response, per_num in zip(origin_responses, PN_number):
            rns = rns_with_average_rankings(response, all_labels)
            selectivity.append(rns[per_num - 1])
            timestep_data.append(np.log2(timestep))
    
    dcg_data_frame = pd.DataFrame(data={'Time Step': timestep_data, 'Selectivity': selectivity})
    print(dcg_data_frame.groupby('Time Step').mean())


def main():
    numerical_time_effect()

if __name__ == '__main__':
    main()



