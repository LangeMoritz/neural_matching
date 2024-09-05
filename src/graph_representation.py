'''Package with functions for creating graph representations of syndromes.'''
import numpy as np

def get_node_list(syndrome_matrix):
    """
    Create two arrays, one containing the syndrome defects,
    and the other containing their corresponding contiguous 
    indices in the matrix representation of the syndrome.
    """
    defect_indices_tuple = np.nonzero(syndrome_matrix)
    defects = syndrome_matrix[defect_indices_tuple]
    return defects, defect_indices_tuple

def get_node_feature_matrix(defects, defect_indices_tuple):
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

    num_node_features = 4 # By default, use 4 node features
        
    # Get defects (non_zero entries), defect indices (indices of defects in flattened syndrome)
    # and defect_indices_tuple (indices in 2D syndrome) of the syndrome matrix
   
    num_defects = defects.shape[0]

    # slicing instead of looping: requires Z defects to be marked with a 3
    defect_indices_tuple = np.transpose(np.array(defect_indices_tuple))

    # get indices of x and z type defects, resp. 
    x_defects = (defects == 1)
    z_defects = (defects == 3)

    # initialize node feature matrix
    node_features = np.zeros([num_defects, num_node_features])
    # defect is x type:
    node_features[x_defects, 0] = 1
    # distance of x tpe defect from northern and western boundary:
    node_features[x_defects, 2:] = defect_indices_tuple[x_defects, :]

    # defect is z type:
    node_features[z_defects, 1] = 1
    # distance of z tpe defect from northern and western boundary:
    node_features[z_defects, 2:] = defect_indices_tuple[z_defects, :]
    
    return node_features

# Function for creating a single graph as a PyG Data object
def get_torch_graph(syndrome, 
    target = None, 
    power = None):
    """
    Creates a (syndrome, target) - pair representing a single
    graph, where each node is connected to m_nearest_nodes neighbouring nodes.
    """
    # get defect indices: 
    defects, defect_indices_tuple = get_node_list(syndrome)
    # Use helper function to create node feature matrix as torch.tensor
    X = get_node_feature_matrix(defects, defect_indices_tuple)

    # set default power of inverted distances to 1
    if power is None:
        power = 1.

    # construct the adjacency matrix!
    n_defects = len(defects)
    y_coord = defect_indices_tuple[0].reshape(n_defects, 1)
    x_coord = defect_indices_tuple[1].reshape(n_defects, 1)
    Adj = np.zeros((n_defects, n_defects))
    Adj = np.maximum(np.abs(y_coord.T - y_coord), np.abs(x_coord.T - x_coord))
    
    np.fill_diagonal(Adj, 1)          # set diagonal elements to nonzero to circumvent division by zero
    Adj = 1./Adj ** power             # scale the edge weights
    np.fill_diagonal(Adj, 0)          # set diagonal elements to zero to exclude self loops
    # # remove all but the m_nearest neighbours
    # if m_nearest_nodes is not None:
    #     # if (np.shape(Adj)[0] + 1) > m_nearest_nodes:
    #     #     Adj = Adj * (Adj >= np.partition(Adj, - m_nearest_nodes, axis = -1)[:,[- m_nearest_nodes]]).astype(int)
    #     for ix, row in enumerate(Adj.T):
    #         # Do not remove edges if a node has (degree <= m)
    #         if np.count_nonzero(row) <= m_nearest_nodes:
    #             continue
    #         # Get indices of all nodes that are not the m nearest
    #         # Remove these edges by setting elements to 0 in adjacency matrix
    #         Adj.T[ix,
    #         np.argpartition(row,-m_nearest_nodes)[:-m_nearest_nodes]] = 0.

    # Adj = np.maximum(Adj, Adj.T) # Make sure for each edge i->j there is edge j->i
    n_edges = np.count_nonzero(Adj) # Get number of edges

    # get the edge indices:
    edge_index = np.nonzero(Adj)
    edge_attr = Adj[edge_index].reshape(n_edges, 1)
    edge_index = np.array(edge_index)

    # set the edge attributes to 1 to test if our method is error-model independent:
    # edge_attr = edge_attr ** 0

    if target is not None:
        # Create torch.tensors of targets
        y = target.reshape(1, 2) 
    else:
        y = None
    return [X.astype(np.float32), edge_index.astype(np.int64,), edge_attr.astype(np.float32), y.astype(np.float32)]
