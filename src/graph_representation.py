'''Package with functions for creating graph representations of syndromes.'''
import numpy as np
import torch
from torch_geometric.data import Data

def get_node_feature_matrix(syndrome, d):
    """
    Creates a node feature matrix of dimensions 
    (number_of_defects, number_of_node_features), where each row
    is the feature vector of a single node.
    The feature vector is defined as
    x = (X, Z, d_north, d_west)
        X: 1 if defect corresponds to a X stabilizer
        Z: 1 if defect corresponds to a Z stabilizer
        d_north: distance to north boundary, i.e. row index in syndrome matrix
        d_west: distance to west boundary, i.e. column index in syndrome matrix
    """

    # syndromes come in shape [x_coordinate, z_coordinate]
    # get the nonzero entries (node features):
    defect_inds = np.nonzero(syndrome)
    node_features = np.transpose(np.array(defect_inds))
    # find the stabilizer types (1: X, 3: Z):
    stabilizer_type = syndrome[defect_inds] == 1
    # add stabilizer type as new node feature ([1, 0]: X, [0, 1]: Z):
    stabilizer_type = stabilizer_type[:, np.newaxis]
    # [shot no., time, space, [stabilizer type]]:
    node_features = np.hstack((node_features, stabilizer_type, ~stabilizer_type)).astype(np.float32)
    # Get defects (non_zero entries), defect indices (indices of defects in flattened syndrome)
    # and defect_indices_tuple (indices in 2D syndrome) of the syndrome matrix
    
    return torch.from_numpy(node_features)

def get_edges(node_features):
    '''Creates edges between all nodes with supremum norm as edge weight'''
    num_nodes = node_features.shape[0]

    # Fully connected graph: Generate edge list
    edge_index = torch.combinations(torch.arange(num_nodes, dtype=torch.int64), r=2).T  # Unique pairs (u, v)
    # Duplicate each edge as (u, v) and (v, u)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # compute the distances between the nodes (start node - end node):
    x_dist = torch.abs(node_features[edge_index[1, :], 0] - 
                    node_features[edge_index[0, :], 0])

    y_dist = torch.abs(node_features[edge_index[1, :], 1] - 
                    node_features[edge_index[0, :], 1])

    # inverse square of the supremum norm between two nodes
    edge_attr = torch.maximum(y_dist, x_dist)
    edge_attr = 1 / edge_attr ** 2

    return edge_index, edge_attr.reshape(-1, 1)

def get_syndrome_graph(code, p):
    """
    Generates a syndrome graph from a given quantum error correction code and error probability.
    Args:
        code (RotatedCode): An instance of the RotatedCode class representing the quantum error correction code.
        p (float): The probability of an error occurring.
    Returns:
        graph: A PyTorch Geometric Data object with node features, edge indices, edge attributes, and the parity of Z errors on the western edge as the label.
               Returns None if there are no X stabilizers in the syndrome.
    """
    d = code.size
    error = code.generate_error_chain(p)
    eq_class = code.get_eq_class(error)
    y = int(eq_class[1]) # the parity of Z errors on the western edge
    syndrome = code.get_syndrome(error)
    # syndrome = np.array([[0, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 1, 0],
    #                      [0, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 0, 0],
    #                      [0, 0, 0, 0, 0, 0]])
    # check number of X stabilizers:
    if np.sum(syndrome == 1) == 0:
        return None
    else:
        node_features = get_node_feature_matrix(syndrome, d)
        edge_index, edge_attr = get_edges(node_features)
        graph = Data(x = node_features, edge_index = edge_index, edge_attr = edge_attr, y = y)
    
        return graph