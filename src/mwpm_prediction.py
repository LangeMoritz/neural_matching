import numpy as np
import networkx as nx
import torch
import itertools

def compute_mwpm_reward(edge_index, edge_weights, num_real_nodes, logical_class):
    """
    Computes the MWPM predictions and returns a reward signal based on the correctness of the correction.

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
    # add boundary nodes (if n_real nodes odd, add an extra virtual node):
    num_boundary_nodes = 2 * num_real_nodes if num_real_nodes % 2 == 0 else 2 * num_real_nodes +1
    # boundary nodes (fully connected)
    boundary_nodes = np.arange(num_real_nodes, num_boundary_nodes + 1)
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
    
    return reward #torch.tensor(reward, dtype=torch.float32)