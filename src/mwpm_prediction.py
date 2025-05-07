import numpy as np
import networkx as nx
from qecsim.graphtools import mwpm, mwpm_multi, mwpm_multi_syndromes
import itertools
import torch 

def compute_mwpm_reward(edge_index, edge_weights, num_real_nodes, num_boundary_nodes, logical_class):
    """
    Computes the MWPM predictions and returns a reward signal based on the correctness of the correction.
    Using the qecsim library for the MWPM algorithm. (blossomV)
    Args:
        edge_weights: The predicted edge weights from the GNN (tensor of size [num_edges]).
        Contains (directed) edges between all real nodes and one edge to both boundaries for
        each real node 
        logical_class: The logical state (0, 1)

    Returns:
        reward: A scalar reward (1 if the correction is correct, - 1 otherwise).
    
    Note: The virtual nodes on the western boundary have indices n, ... 2*n - 1
    """
    edge_index = edge_index.cpu().detach().numpy()
    edge_weights = edge_weights.cpu().detach().numpy()

    # Add edges with weights using a generator
    edges = {tuple(x): w for x, w in zip(edge_index.T, edge_weights)}

    # add boundary nodes (if n_real nodes odd, add an extra virtual node), (fully connected)
    boundary_nodes = np.arange(num_real_nodes, num_boundary_nodes + num_real_nodes)
    # Add edges of weight 0 between all pairs of boundary nodes
    edges.update({(u, v): 0 for u, v in itertools.combinations(boundary_nodes, 2)})

    # do the MWPM:
    matching = mwpm(edges)
    matching_array = np.array(list(matching))

    # Create boolean masks for the first and second sets
    mask_u_in_set1 = (matching_array[:, 0] >= 0) & (matching_array[:, 0] < num_real_nodes)
    mask_v_in_set2 = (matching_array[:, 1] >= num_real_nodes) & (matching_array[:, 1] < 2 * num_real_nodes)

    mask_v_in_set1 = (matching_array[:, 1] >= 0) & (matching_array[:, 1] < num_real_nodes)
    mask_u_in_set2 = (matching_array[:, 0] >= num_real_nodes) & (matching_array[:, 0] < 2 * num_real_nodes)

    # Combine both conditions: (u in set1 and v in set2) or (v in set1 and u in set2)
    valid_edges_mask = (mask_u_in_set1 & mask_v_in_set2) | (mask_v_in_set1 & mask_u_in_set2)

    # Count the edges running through the left boundary of the code:
    num_left_edges = np.sum(valid_edges_mask)
    
    # get the predicted logical state:
    predicted_state = num_left_edges % 2

    # compare with logical state:
    correct_prediction = (predicted_state == logical_class)
    # Assign reward
    reward = 1 if correct_prediction else -1
    
    return reward

def compute_mwpm_rewards_multi_syndrome(syndrome_graphs, num_real_nodes_list, num_boundary_nodes_list, logical_classes, num_draws_per_sample, num_samples_per_epoch, scale_max=10000):
    """
    Args:
        syndrome_graphs: list of (edge_index, edge_weights_draws), where 
        edge_weights_draws has shape (num_draws, num_edges) and edge_index has shape (2, num_edges).
        num_real_nodes_list: list of integers
        num_boundary_nodes_list: list of integers
        logical_classes: list of integers (0 or 1)
        num_draws_per_sample: number of draws per sample
        num_samples_per_epoch: number of samples per epoch
        scale_max: maximum value for scaling the edge weights to integers

    Returns:
        rewards: np.ndarray of shape (num_draws, num_syndromes)
    """
    all_graphs = []
    for (edge_index, edge_weights_draws), num_real_nodes, num_boundary_nodes in zip(syndrome_graphs, num_real_nodes_list, num_boundary_nodes_list):
        # Assume edge_index: shape (2, num_edges), edge_weights_draws: (num_draws, num_edges)

        # 3. Generate boundary edges between virtual nodes
        boundary_nodes = np.arange(num_real_nodes, num_real_nodes + num_boundary_nodes)
        boundary_pairs = np.array(list(itertools.combinations(boundary_nodes, 2)), dtype=np.int32)

        # 4. Concatenate all edges
        all_edges = np.vstack([edge_index.T, boundary_pairs])  # shape (num_total_edges, 2)

        # 5. Final edge_index: shape (2, num_total_edges)
        edge_index_full = all_edges.T

        # number of boundary edges:
        num_boundary_edges = edge_index_full.shape[1] - edge_weights_draws.shape[1]
        # Extend weights with zeros for boundary edges
        zero_pad = np.zeros((num_draws_per_sample, num_boundary_edges), dtype=edge_weights_draws.dtype)
        # shape: (num_draws, num_total_edges)
        full_weights = np.hstack([edge_weights_draws, zero_pad])
        all_graphs.append((edge_index_full, full_weights))

    matches = mwpm_multi_syndromes(all_graphs, draws_per_syndrome = num_draws_per_sample)  # shape: (n_syndromes, n_draws, n_nodes)
    
    rewards = torch.ones((num_draws_per_sample, num_samples_per_epoch), dtype=torch.float32)
    for s in range(num_samples_per_epoch):
        match = matches[s]  # shape (n_draws, n_nodes)
        n_draws, n_nodes = match.shape
        n_real = num_real_nodes_list[s]
        log_class = logical_classes[s]

        # note: because match is symmetric, we only need to check one side:
        u = np.arange(n_nodes)
        u_broadcast = np.broadcast_to(u, (n_draws, n_nodes))
        v = match  # shape (n_draws, n_nodes)

        # check if u is a real node:
        mask_u = (u_broadcast >= 0) & (u_broadcast < n_real)
        # check if v is a virtual node on the left boundary:
        mask_v = (v >= n_real) & (v < 2 * n_real)

        # count number of edges between real and left boundary nodes:
        num_left_edges = np.count_nonzero((mask_u & mask_v), axis=1)
        predicted_wrong = ((num_left_edges % 2) != log_class)
        rewards[predicted_wrong, s] = - 1.0

    return rewards

def compute_mwpm_rewards_multiple_draws(edge_index, edge_weights_draws, num_real_nodes, num_boundary_nodes, logical_class):
    """
    Computes MWPM predictions for multiple edge weight draws and returns reward signals.

    Args:
        edge_index: Tensor of shape [2, num_edges], edge indices.
        edge_weights_draws: Tensor of shape [num_draws, num_edges], predicted weights per draw.
        num_real_nodes: Number of real nodes.
        num_boundary_nodes: Number of boundary (virtual) nodes.
        logical_class: The true logical state (0 or 1).

    Returns:
        rewards: List of scalar rewards for each draw (1 if correct, -1 otherwise).
    """
    edge_index = edge_index.cpu().detach().numpy()
    edge_weights_draws = edge_weights_draws.cpu().detach().numpy()  # shape: [num_draws, num_edges]

    # Construct base graph (structure only)
    edges = list(map(tuple, edge_index.T))
    base_graph = {e: 0 for e in edges}  # use dummy weight for structure

    # Add edges of weight 0 between all pairs of boundary nodes
    boundary_nodes = np.arange(num_real_nodes, num_real_nodes + num_boundary_nodes)
    for u, v in itertools.combinations(boundary_nodes, 2):
        base_graph[(u, v)] = 0

    # Expand each draw with 0-weights for the boundary node edges
    n_draws = edge_weights_draws.shape[0]
    num_edge_main = edge_weights_draws.shape[1]
    num_boundary_edges = len(base_graph) - num_edge_main
    zero_weights = np.zeros((n_draws, num_boundary_edges), dtype=edge_weights_draws.dtype)
    full_draws = np.hstack([edge_weights_draws, zero_weights])

    # Compute matchings using mwpm_multi
    matchings = mwpm_multi(base_graph, full_draws)

    rewards = []
    for matching in matchings:
        matching_array = np.array(list(matching))

        mask_u_in_set1 = (matching_array[:, 0] >= 0) & (matching_array[:, 0] < num_real_nodes)
        mask_v_in_set2 = (matching_array[:, 1] >= num_real_nodes) & (matching_array[:, 1] < 2 * num_real_nodes)
        mask_v_in_set1 = (matching_array[:, 1] >= 0) & (matching_array[:, 1] < num_real_nodes)
        mask_u_in_set2 = (matching_array[:, 0] >= num_real_nodes) & (matching_array[:, 0] < 2 * num_real_nodes)

        valid_edges_mask = (mask_u_in_set1 & mask_v_in_set2) | (mask_v_in_set1 & mask_u_in_set2)
        num_left_edges = np.sum(valid_edges_mask)

        predicted_state = num_left_edges % 2
        reward = 1 if predicted_state == logical_class else -1
        rewards.append(reward)

    return rewards


def compute_mwpm_reward_parallel(edge_index, edge_weights, num_real_nodes, num_boundary_nodes, logical_class):
    """
    Computes the MWPM predictions and returns a reward signal based on the correctness of the correction.
    Using the qecsim library for the MWPM algorithm. (blossomV)
    Args:
        edge_weights: The predicted edge weights from the GNN (numpy array of size [num_edges]).
        Contains (directed) edges between all real nodes and one edge to both boundaries for
        each real node 
        logical_class: The logical state (0, 1)

    Returns:
        reward: A scalar reward (1 if the correction is correct, - 1 otherwise).
    
    Note: The virtual nodes on the western boundary have indices n, ... 2*n - 1
    """

    # Add edges with weights using a generator
    edges = {tuple(x): w for x, w in zip(edge_index.T, edge_weights)}

    # add boundary nodes (if n_real nodes odd, add an extra virtual node), (fully connected)
    boundary_nodes = np.arange(num_real_nodes, num_boundary_nodes + num_real_nodes)
    # Add edges of weight 0 between all pairs of boundary nodes
    edges.update({(u, v): 0 for u, v in itertools.combinations(boundary_nodes, 2)})

    # do the MWPM:
    matching = mwpm(edges)
    matching_array = np.array(list(matching))

    # Create boolean masks for the first and second sets
    mask_u_in_set1 = (matching_array[:, 0] >= 0) & (matching_array[:, 0] < num_real_nodes)
    mask_v_in_set2 = (matching_array[:, 1] >= num_real_nodes) & (matching_array[:, 1] < 2 * num_real_nodes)

    mask_v_in_set1 = (matching_array[:, 1] >= 0) & (matching_array[:, 1] < num_real_nodes)
    mask_u_in_set2 = (matching_array[:, 0] >= num_real_nodes) & (matching_array[:, 0] < 2 * num_real_nodes)

    # Combine both conditions: (u in set1 and v in set2) or (v in set1 and u in set2)
    valid_edges_mask = (mask_u_in_set1 & mask_v_in_set2) | (mask_v_in_set1 & mask_u_in_set2)

    # Count the edges running through the left boundary of the code:
    num_left_edges = np.sum(valid_edges_mask)
    
    # get the predicted logical state:
    predicted_state = num_left_edges % 2

    # compare with logical state:
    correct_prediction = (predicted_state == logical_class)
    # Assign reward
    reward = 1 if correct_prediction else -1
    
    return reward

def compute_mwpm_reward_networkx(edge_index, edge_weights, num_real_nodes, num_boundary_nodes, logical_class):
    """
    Computes the MWPM predictions and returns a reward signal based on the correctness of the correction.
    Using the NetworkX library for the MWPM algorithm.

    Args:
        edge_weights: The predicted edge weights from the GNN (tensor of size [num_edges]).
        Contains (directed) edges between all real nodes and one edge to both boundaries for
        each real node 
        logical_class: The logical state (0, 1)

    Returns:
        reward: A scalar reward (1 if the correction is correct, - 1 otherwise).
    
    Note: The virtual nodes on the western boundary have indices n, ... 2*n - 1
    """
    edge_index = edge_index.detach().numpy()
    edge_weights = edge_weights.detach().numpy()
    G = nx.Graph()

    # Add edges with weights using a generator
    G.add_edges_from((u, v, {"weight": -w}) for u, v, w in zip(edge_index[0], edge_index[1], edge_weights))
    # add boundary nodes (if n_real nodes odd, add an extra virtual node), (fully connected)
    boundary_nodes = np.arange(num_real_nodes, num_boundary_nodes + num_real_nodes)
    # Add edges of weight 0 between all pairs of boundary nodes
    G.add_edges_from((u, v, {"weight": 0}) for u, v in itertools.combinations(boundary_nodes, 2))

    # do the MWPM:
    matching = nx.algorithms.max_weight_matching(G, maxcardinality=True)
    matching_array = np.array(list(matching))

    # Create boolean masks for the first and second sets
    mask_u_in_set1 = (matching_array[:, 0] >= 0) & (matching_array[:, 0] < num_real_nodes)
    mask_v_in_set2 = (matching_array[:, 1] >= num_real_nodes) & (matching_array[:, 1] < 2 * num_real_nodes)

    mask_v_in_set1 = (matching_array[:, 1] >= 0) & (matching_array[:, 1] < num_real_nodes)
    mask_u_in_set2 = (matching_array[:, 0] >= num_real_nodes) & (matching_array[:, 0] < 2 * num_real_nodes)

    # Combine both conditions: (u in set1 and v in set2) or (v in set1 and u in set2)
    valid_edges_mask = (mask_u_in_set1 & mask_v_in_set2) | (mask_v_in_set1 & mask_u_in_set2)

    # Count the edges running through the left boundary of the code:
    num_left_edges = np.sum(valid_edges_mask)
    
    # get the predicted logical state:
    predicted_state = num_left_edges % 2

    # compare with logical state:
    correct_prediction = (predicted_state == logical_class)
    # Assign reward
    reward = 1 if correct_prediction else -1
    
    return reward
