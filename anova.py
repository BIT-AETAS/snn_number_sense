# -*- encoding: utf-8 -*-
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from scipy import stats
from scipy.optimize import curve_fit

from statsmodels.formula.api import ols       # Ordinary Least Squares fitting
from statsmodels.stats.anova import anova_lm  # Analysis of Variance

class AnovaRunner(object):
    def __init__(self, p1, p2, p3, numerositys, controls, responses):
        """
        Initializes the object with provided parameters and prepares response data.

        Args:
            p1: p_value 1.
            p2: p_value 2.
            p3: p_value 3.
            numerositys: numerosity associated with the data.
            controls: Control conditions.
            responses: Spiking response data, expected as a NumPy array with shape (data_size, neuron_number).

        Attributes:
            PN_index (list): List to store PN indices.
            p1, p2, p3: Stored parameters.
            numerositys: Stored numerositys.
            controls: Stored controls.
            responses: Spiking response data.
            neuron_number (int): Number of neurons, inferred from responses.
            all_data: Placeholder for all data, initialized as None.
        """
        self.PN_index = []
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.numerositys = numerositys
        self.controls = controls
        # Convert shape from data_size * neuron_number to neuron_number * data_size
        self.responses = responses
        self.neuron_number = self.responses.shape[1]
        self.all_data = None

    def anova_anlyze(self, index, df):
        """
        Performs ANOVA analysis on the provided DataFrame and appends the index to PN_index if specific p-value criteria are met.

        Args:
            index (int): The index to potentially append to PN_index.
            df (pd.DataFrame): The DataFrame containing the data for ANOVA analysis. Must include columns 'output', 'number', and 'control'.

        Returns:
            None

        Notes:
            - This method is slow and recommended only for small datasets. For large datasets, consider using the overloaded method below.
            - Fits an OLS model with 'output' as the dependent variable and 'number', 'control', and their interaction as independent variables.
            - Performs ANOVA on the fitted model.
            - Appends the index to PN_index if:
            - p-value for 'number' < self.p1
            - p-value for 'control' > self.p2
            - p-value for 'number:control' > self.p3
        """
        ols_model = ols(
            'output ~ C(number) + C(control) + C(number)*C(control)', data=df).fit()
        anovat = anova_lm(ols_model)
        if anovat['PR(>F)']['C(number)'] < self.p1 and anovat['PR(>F)']['C(control)'] > self.p2 and anovat['PR(>F)']['C(number):C(control)'] > self.p3:
            self.PN_index.append(index)

    def run(self):
        """
        Executes a two-way ANOVA analysis on the spike responses.

        This method performs the following steps:
        1. Identifies active neurons based on their response values.
        2. Transposes the response data to align neuron outputs.
        3. Iterates over each active neuron, creating a DataFrame with corresponding numerositys, controls, and outputs.
        4. Runs the ANOVA analysis for each active neuron using the provided DataFrame.
        5. Prints the number and rate of active neurons.

        Assumes the existence of the following instance attributes:
            - self.responses: np.ndarray, spike response data.
            - self.numerositys: array-like, numerositys for each sample.
            - self.controls: array-like, control conditions for each sample.
            - self.anova_anlyze: method to perform ANOVA analysis.
            - self.neuron_number: int, total number of neurons.

        Prints progress and summary statistics to the console.
        """
        start_time = time.time()
        print('Start 2 ways anova anlyze: ', time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
        # Only test active cells
        active_cells = np.where(np.abs(self.responses).max(axis=0) > 0)[0]
        activate_neuron_output = np.transpose(self.responses[:, active_cells], [1, 0])
        for i in tqdm([range(len(active_cells))]):
            df = pd.DataFrame(
                {'number': self.numerositys, 'control': self.controls, 'output': activate_neuron_output[i]})
            self.anova_anlyze(active_cells[i], df)
        end_time = time.time()
        print('End 2 ways anova anlyze: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
        print('active neuron number: %d, active neuron rate: %f' % (len(active_cells), 
                                                                    len(active_cells) / self.neuron_number))

    def get_preferred_numerosity_and_tuning_curves(self, id):
        """
        Computes the tuning curves and preferred numerosity for a specified neuron.

        Args:
            id (int): The index of the neuron for which to compute the tuning curves.

        Returns:
            PN_outputs_0 (np.ndarray): Mean responses of the neuron for each numerosity when control is 0.
            PN_outputs_1 (np.ndarray): Mean responses of the neuron for each numerosity when control is 1.
            PN_outputs_2 (np.ndarray): Mean responses of the neuron for each numerosity when control is 2.
            avg_output_df (np.ndarray): Average response across all control conditions for each numerosity.
            per_num (int): The preferred numerosity.
        """
        numerosity_array = np.sort(np.unique(self.numerositys))
        neuron2output = np.transpose(self.responses, [1, 0])
        PN_outputs_0 = np.array([neuron2output[id][(self.controls == 0) & (
            self.numerositys == i)].mean() for i in numerosity_array])
        PN_outputs_1 = np.array([neuron2output[id][(self.controls == 1) & (
            self.numerositys == i)].mean() for i in numerosity_array])
        PN_outputs_2 = np.array([neuron2output[id][(self.controls == 2) & (
            self.numerositys == i)].mean() for i in numerosity_array])
        avg_output_df = (PN_outputs_0 + PN_outputs_1 + PN_outputs_2) / 3
        per_num = int(numerosity_array[np.argmax(avg_output_df, axis=0)])
        return PN_outputs_0, PN_outputs_1, PN_outputs_2, avg_output_df, per_num

    def get_weber_fechner(self,):

        # Gaussian function
        def gauss(x, a, miu, sigma):
            return a * np.exp(-(x - miu)**2 / (2 * sigma**2))

        # Perform Gaussian fitting
        def gauss_fit(nonzero_idx, x, tuning_curves):
            miu = []
            sigma = []
            a = []
            for j in nonzero_idx:
                popt, pcov = curve_fit(gauss, x, tuning_curves[j-1], bounds=(0, [1.5, 30., 100.]))
                a.append(popt[0])
                miu.append(popt[1])
                sigma.append(popt[2])
            return a, miu, sigma

        # Calculate R^2 using mathematical formula
        def getR2(nonzero_idx, x, tuning_curves, a, miu, sigma):
            R2 = []
            for idx, number in enumerate(nonzero_idx):
                ymean = np.mean(tuning_curves[number-1])
                ygauss = gauss(x, a[idx], miu[idx], sigma[idx])
                ss_total = np.sum((tuning_curves[number-1] - ymean)**2)
                ss_res = np.sum((tuning_curves[number-1] - ygauss)**2)
                R2.append(1 - ss_res/ss_total)

            return R2

        # Remove possible nan and inf values
        def clean_nan_inf(x):
            for i, _ in enumerate(x):
                if np.isnan(x[i]) == True:
                    x[i] = 0
                if np.isinf(x[i]) == True:
                    x[i] = 1
            return x

        # Sum all neuron-related data corresponding to the same PN and assign to sum_avg_t
        tuning_curves = np.zeros((30, 30))
        for n_id in self.PN_index:
            PN_resonses_0, PN_resonses_1, PN_resonses_2, avg_resonses, pn = self.get_preferred_numerosity_and_tuning_curves(n_id)
            tuning_curves[pn - 1] = np.add(tuning_curves[pn - 1], avg_resonses).tolist()
        nonzero_idx = []                                                               # Store existing PN
        x = range(1, 31)

        for i in range(30):
            if (sum(tuning_curves[i])) == 0:
                continue
            nonzero_idx.append(i+1)

            # Normalize tuning_curves[i]
            PN_avg_max = max(tuning_curves[i])
            PN_avg_min = min(tuning_curves[i])
            PN_avg = PN_avg_max - PN_avg_min
            tuning_curves[i] = [(k - PN_avg_min) / PN_avg for k in list(tuning_curves[i])]
            tuning_curves[i] = clean_nan_inf(tuning_curves[i])

        
        # Calculate and plot the mean R^2 for x, x^1/2, x^1/3, log2(x)
        a_linear, miu_linear, sigma_linear = gauss_fit(nonzero_idx, x, tuning_curves)
        R2_linear = getR2(nonzero_idx, x, tuning_curves, a_linear, miu_linear, sigma_linear)
        a_P2, miu_P2, sigma_P2 = gauss_fit(nonzero_idx, np.sqrt(x), tuning_curves)
        R2_P2 = getR2(nonzero_idx, np.sqrt(x), tuning_curves, a_P2, miu_P2, sigma_P2)
        x_P3 = map(lambda x: x**(1/3), x)
        x_P3 = [i for i in x_P3]
        a_P3, miu_P3, sigma_P3 = gauss_fit(nonzero_idx, x_P3, tuning_curves)
        R2_P3 = getR2(nonzero_idx, x_P3, tuning_curves, a_P3, miu_P3, sigma_P3)
        a_log, miu_log, sigma_log = gauss_fit(nonzero_idx, np.log2(x), tuning_curves)
        R2_log = getR2(nonzero_idx, np.log2(x), tuning_curves, a_log, miu_log, sigma_log)

        print("R2_linear: ", np.mean(R2_linear))
        print("R2_P2: ", np.mean(R2_P2))
        print("R2_P3: ", np.mean(R2_P3))
        print("R2_log: ", np.mean(R2_log))


class FastAnovaRunner(AnovaRunner):
    def __init__(self, p1, p2, p3, numerositys, controls, responses):
        super(FastAnovaRunner, self).__init__(
            p1, p2, p3, numerositys, controls, responses)
        
    def anova_two_way(self, A, B, Y):
        num_cells = Y.shape[1]
        
        # Find active neurons
        active_cells = np.where(np.abs(Y).max(axis=0) > 0)[0]
        print("active neuron number: %d, active neuron rate: %f" %
                (len(active_cells), len(active_cells) / num_cells))
        
        # Get levels of factors A and B
        A_levels = np.unique(A)
        a = len(A_levels)
        B_levels = np.unique(B)
        b = len(B_levels)
        
        # Initialize result arrays
        pA = np.ones(num_cells)
        pB = np.ones(num_cells)
        pAB = np.ones(num_cells)
        
        # Only process active neurons
        Y_active = Y[:, active_cells]
        
        # Calculate grand mean
        grand_mean = np.mean(Y_active, axis=0)
        
        # Initialize group means, sample counts, and sum of squares
        means_A = np.zeros((a, len(active_cells)))
        means_B = np.zeros((b, len(active_cells)))
        means_AB = np.zeros((a, b, len(active_cells)))
        n_AB = np.zeros((a, b), dtype=int)
        
        # Calculate group means and sample counts
        for i, a_level in enumerate(A_levels):
            A_mask = (A == a_level)
            means_A[i] = np.mean(Y_active[A_mask], axis=0)
            
            for j, b_level in enumerate(B_levels):
                B_mask = (B == b_level)
                AB_mask = A_mask & B_mask
                cell_n = np.sum(AB_mask)
                n_AB[i, j] = cell_n
                
                if cell_n > 0:
                    means_AB[i, j] = np.mean(Y_active[AB_mask], axis=0)
        
        for j, b_level in enumerate(B_levels):
            B_mask = (B == b_level)
            means_B[j] = np.mean(Y_active[B_mask], axis=0)
        
        # Total sample count
        N = np.sum(n_AB)
        
        # Calculate sum of squares (Type III method)
        SSA = np.zeros(len(active_cells))
        SSB = np.zeros(len(active_cells))
        SSAB = np.zeros(len(active_cells))
        SSE = np.zeros(len(active_cells))
        SST = np.sum((Y_active - grand_mean)**2, axis=0)
        
        # Calculate predicted values for each observation
        for i, a_level in enumerate(A_levels):
            A_mask = (A == a_level)
            
            for j, b_level in enumerate(B_levels):
                B_mask = (B == b_level)
                AB_mask = A_mask & B_mask
                cell_n = np.sum(AB_mask)
                
                if cell_n > 0:
                    # Type III sum of squares
                    effect_A = means_A[i] - grand_mean
                    effect_B = means_B[j] - grand_mean
                    effect_AB = means_AB[i, j] - means_A[i] - means_B[j] + grand_mean
                    
                    SSA += cell_n * effect_A**2
                    SSB += cell_n * effect_B**2
                    SSAB += cell_n * effect_AB**2
                    
                    # Calculate error sum of squares
                    cell_values = Y_active[AB_mask]
                    cell_mean = means_AB[i, j]
                    SSE += np.sum((cell_values - cell_mean.reshape(1, -1))**2, axis=0)
        
        # Calculate degrees of freedom
        DFA = a - 1
        DFB = b - 1
        DFAB = DFA * DFB
        DFE = N - (a * b)
        
        # Calculate mean squares
        MSA = SSA / DFA
        MSB = SSB / DFB
        MSAB = SSAB / DFAB
        MSE = SSE / DFE
        
        # Calculate F statistics
        FA = MSA / MSE
        FB = MSB / MSE
        FAB = MSAB / MSE
        
        # Calculate p-values
        pA_active = stats.f.sf(FA, DFA, DFE)
        pB_active = stats.f.sf(FB, DFB, DFE)
        pAB_active = stats.f.sf(FAB, DFAB, DFE)
        
        # Assign results to corresponding neurons
        pA[active_cells] = pA_active
        pB[active_cells] = pB_active
        pAB[active_cells] = pAB_active
        
        return pA, pB, pAB

    def run(self):
        start_time = time.time()
        print("Start two-ways anova: ", time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
        pN, pC, pNC = self.anova_two_way(
            self.numerositys, self.controls, self.responses)
        self.PN_index = np.where((pN < self.p1) & (pNC > self.p2) & (pC > self.p3))[0]
        end_time = time.time()
        print("End two-ways anova: ", time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(end_time)))

