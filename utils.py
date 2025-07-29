# -*- encoding: utf-8 -*-
import numpy as np

def get_preferred_numerosity(responses, labels, controls, indexs):
    """
    Calculates the preferred numerosity (PN) for each specified index based on neural responses under different control conditions.

    Args:
        responses (np.ndarray): Array of neural responses, where each entry corresponds to a neuron's response across trials.
        labels (np.ndarray): Array of numerosity labels for each trial.
        controls (np.ndarray): Array indicating the control condition for each trial (e.g., 0, 1, 2).
        indexs (iterable): Indices of neurons (or response units) to analyze.

    Returns:
        tuple:
            avg_responses (np.ndarray): Averaged responses for each neuron across all control conditions and numerosity labels.
            origin_responses (np.ndarray): Original responses for each neuron.
            pn_list (np.ndarray): Preferred numerosity (label with maximum average response) for each neuron.
    """
    label_number = np.sort(np.unique(labels))
    pn_list = []
    avg_responses = []
    origin_responses = []
    for index in indexs:
        PN_outputs_0 = np.array([responses[index][(controls == 0) & (labels == i)].mean() for i in label_number])
        PN_outputs_1 = np.array([responses[index][(controls == 1) & (labels == i)].mean() for i in label_number])
        PN_outputs_2 = np.array([responses[index][(controls == 2) & (labels == i)].mean() for i in label_number])
        avg_output_df = (PN_outputs_0 + PN_outputs_1 + PN_outputs_2) / 3
        pn = int(label_number[np.argmax(avg_output_df, axis=0)])
        pn_list.append(pn)
        avg_responses.append(avg_output_df)
        origin_responses.append(responses[index])
    return np.array(avg_responses), np.array(origin_responses), np.array(pn_list)


def assign_average_rankings(scores):
    """
    Assign average ranks to tied scores using the average ranking method.

    Args:
        scores (np.ndarray): Scores of documents in descending order. Tied scores indicate tied ranks.

    Returns:
        np.ndarray: Array of average rank rankings (1-based) for each score.
    """
    n = len(scores)
    if n == 0:
        return np.array([])

    differences = np.diff(scores)
    tie_starts = np.concatenate(([0], np.where(differences != 0)[0] + 1))
    tie_ends = np.concatenate((tie_starts[1:] - 1, [n - 1]))

    avg_ranks = np.zeros(n, dtype=float)
    for start, end in zip(tie_starts, tie_ends):
        avg_rank = (start + 1 + end + 1) / 2.0
        avg_ranks[start:end+1] = avg_rank

    return avg_ranks


def rns_with_average_rankings(response, labels):
    """
    Compute Rank-based Numerical Selectivity (RNS) for each unique label.

    Args:
        response (np.ndarray): Prediction scores for each sample.
        labels (np.ndarray): Numerosity for each sample.

    Returns:
        list: RNS value for each unique numerosity.
    """
    label_number = np.sort(np.unique(labels))
    sorted_indices = np.argsort(-response)
    sorted_labels = np.array(labels[sorted_indices])
    sorted_response = response[sorted_indices]
    avg_rankings = assign_average_rankings(sorted_response)
    sg_label = []
    for label in label_number:
        # SG for current numerosity
        sg = np.sum(1 / np.log2(avg_rankings[sorted_labels == label] + 1))
        # ESG for current numerosity
        num_label = np.sum(sorted_labels == label)
        if num_label == 0:
            esg = 1.0  # Avoid division by zero
        else:
            esg = np.sum(1 / np.log2(np.arange(1, num_label + 1) + 1))
        sg_label.append(sg / esg if esg > 0 else 0.0)
    return sg_label


def average_precision(predictions, labels, targets, k=None):
    """
    Compute average precision for each target label.

    Args:
        predictions (np.ndarray): Prediction scores for each sample.
        labels (np.ndarray): Ground truth labels for each sample.
        targets (iterable): Target labels to compute average precision for.
        k (int, optional): Number of top predictions to consider. Defaults to len(labels).

    Returns:
        np.ndarray: Average precision for each target label.
    """
    if k is None:
        k = len(labels)

    average_precisions = []

    sorted_indices = np.argsort(-predictions)
    sorted_labels = labels[sorted_indices[:k]]
    for target in targets:
        # Calculate average precision for the target label
        hit_labels = (sorted_labels == target).astype(int)
        cumulative_hits = np.cumsum(hit_labels)
        precision_at_k = cumulative_hits / np.arange(1, k + 1)
        avg_precision = np.sum(precision_at_k) / k
        average_precisions.append(avg_precision)
    return np.array(average_precisions)


def get_PN_connection(PN_units, layer_start_index, last_layer_start_index, model_weight, kernel_info):
    k_h = kernel_info['k_h']
    k_w = kernel_info['k_w']
    s_h = kernel_info['s_h']
    s_w = kernel_info['s_w']
    p_h = kernel_info['p_h']
    p_w = kernel_info['p_w']
    d_h = kernel_info['d_h']
    d_w = kernel_info['d_w']
    in_channel = kernel_info['in_channel']
    out_channel = kernel_info['out_channel']
    output_h = kernel_info['output_h']
    output_w = kernel_info['output_w']
    input_h = kernel_info['input_h']
    input_w = kernel_info['input_w']
    PN_connections = []

    for PN_index in PN_units:
        PN_connection = []
        index = PN_index - layer_start_index
        x = (index // out_channel) // output_w
        y = (index // out_channel) % output_w
        channel = index % out_channel
        for h in range(k_h):
            for w in range(k_w):
                input_x = x * s_h - p_h + h * d_h
                input_y = y * s_w - p_w + w * d_w
                if input_x >= 0 and input_x < input_h and input_y >= 0 and input_y < input_w:
                    for c in range(in_channel):
                        input_index = input_x * input_w * in_channel + input_y * in_channel + c
                        PN_connection.append([PN_index, input_index + last_layer_start_index, model_weight[channel, c, h, w].item()])
        PN_connections.append(PN_connection)
    return PN_connections


def get_PN_connection_with_pooling(PN_units, layer_start_index, last_layer_start_index, model_weight, kernel_info):
    """
    Generates the connection list for Projection Neurons (PN) with pooling, based on convolution and pooling parameters.

    Args:
        PN_units (list or iterable): Indices of the PN units in the current layer.
        layer_start_index (int): The starting index of the current layer's units.
        last_layer_start_index (int): The starting index of the previous layer's units.
        model_weight (torch.Tensor or numpy.ndarray): The weight tensor of the convolutional layer, 
            typically of shape (out_channel, in_channel, kernel_height, kernel_width).
        kernel_info (dict): Dictionary containing convolution and pooling parameters:
            - 'k_h', 'k_w': Kernel height and width.
            - 's_h', 's_w': Stride height and width.
            - 'p_h', 'p_w': Padding height and width.
            - 'd_h', 'd_w': Dilation height and width.
            - 'in_channel': Number of input channels.
            - 'out_channel': Number of output channels.
            - 'output_h', 'output_w': Output feature map height and width.
            - 'input_h', 'input_w': Input feature map height and width.
            - 'pool_kernel_size': Pooling kernel size (assumed square).
            - 'pool_stride': Pooling stride.
            - 'input_pool_h', 'input_pool_w': Height and width of the input to the pooling layer.

    Returns:
        list: A list of lists, where each inner list contains the connections for a PN unit.
            Each connection is represented as [PN_index, input_index, weight], where:
                - PN_index: Index of the PN unit in the current layer.
                - input_index: Index of the connected unit in the previous layer (after pooling).
                - weight: Weight of the connection.
    """
    k_h = kernel_info['k_h']
    k_w = kernel_info['k_w']
    s_h = kernel_info['s_h']
    s_w = kernel_info['s_w']
    p_h = kernel_info['p_h']
    p_w = kernel_info['p_w']
    d_h = kernel_info['d_h']
    d_w = kernel_info['d_w']
    in_channel = kernel_info['in_channel']
    out_channel = kernel_info['out_channel']
    output_h = kernel_info['output_h']
    output_w = kernel_info['output_w']
    input_h = kernel_info['input_h']
    input_w = kernel_info['input_w']
    pool_kernel_size = kernel_info['pool_kernel_size']
    pool_stride = kernel_info['pool_stride']
    input_pool_h = kernel_info['input_pool_h']
    input_pool_w = kernel_info['input_pool_w']
    PN_connections = []

    for PN_index in PN_units:
        PN_connection = []
        PN_connect_prelayer = []
        index = PN_index - layer_start_index
        x = (index // out_channel) // output_w
        y = (index // out_channel) % output_w
        channel = index % out_channel
        for h in range(k_h):
            for w in range(k_w):
                input_x = x * s_h - p_h + h * d_h
                input_y = y * s_w - p_w + w * d_w
                if input_x >= 0 and input_x < input_h and input_y >= 0 and input_y < input_w:
                    for c in range(in_channel):
                        PN_connect_prelayer.append([input_x, input_y, c, model_weight[channel, c, h, w].item()])
        for prelayer_unit in PN_connect_prelayer:
            x, y, channel, weight = prelayer_unit
            for h in range(pool_kernel_size):
                for w in range(pool_kernel_size):
                    input_x = x * pool_stride + h
                    input_y = y * pool_stride + w
                    if input_x >= 0 and input_x < input_pool_h and input_y >= 0 and input_y < input_pool_w:
                        input_index = input_x * input_pool_w * in_channel + input_y * in_channel + channel
                        PN_connection.append([PN_index, input_index + last_layer_start_index, weight])
        PN_connections.append(PN_connection)
    return PN_connections