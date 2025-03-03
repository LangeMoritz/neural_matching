import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class EdgeWeightGNN(nn.Module):
    '''
    A Graph Neural Network (GNN) for predicting edge weights based on node features and edge embeddings.
    This model uses graph convolution layers (GCNs) for node feature transformation and an MLP for edge embedding
    to predict the edge weights means of a Gaussian distribution.

    Attributes:
        graph_conv_layers (nn.ModuleList): A list of GraphConv layers applied sequentially to update node features.
        edge_mlp (nn.Sequential): A multi-layer perceptron (MLP) for processing edge embeddings to predict edge weights.
    Args:
        node_feat_dim (int): The dimension of the input node features (e.g., feature size for each node).
        hidden_dim (int): The number of hidden units in each graph convolution layer.
        num_gcn_layers (int): The number of graph convolution layers to be stacked in the model.
    '''

    def __init__(self, node_feat_dim = 4, hidden_dim = 64, num_gcn_layers = 2):
        super(EdgeWeightGNN, self).__init__()

        # GCN layers
        self.graph_conv_layers = nn.ModuleList()
        self.graph_conv_layers.append(GraphConv(node_feat_dim, hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.graph_conv_layers.append(GraphConv(hidden_dim, hidden_dim))

        # Edge embedding MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))  # Output scalar edge weight

    def get_device(self):
        return next(self.parameters()).device
    
    def forward(self, x, edge_index, edge_weights):
        """
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, node_feat_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            edge_weights (torch.Tensor): Edge features of shape [num_edges, 1].
        Returns:
            edge_weights (torch.Tensor): Predicted edge weights of shape [num_edges].
        """
        # get the device the model is running at:
        self.device = self.get_device()

        # store the stabilizer type of the nodes
        stabilizer_type = x[:, -2:]
        # Step 1: Apply GCN layers to update node embeddings
        for graph_conv in self.graph_conv_layers:
            x = F.relu(graph_conv(x, edge_index, edge_weights))

        # Step 2: Only keep X type stabilizers:
        x, edge_index, edge_weights = self.filter_nodes_and_edges(x, stabilizer_type, edge_index, edge_weights)

        # Step 3: Remove duplicates (need only to process (source, target)):
        edge_index, edge_weights = self.remove_duplicates(edge_index, edge_weights)
        
        # Step 4: Add boundary edges connecting each node to two virtual nodes
        edge_index, edge_weights, x, num_real_nodes = self.add_boundary_edges(
                                                    edge_index, edge_weights, x)
        # add boundary nodes (if n_real nodes odd, add an extra virtual node):
        if num_real_nodes % 2 == 0:
            num_boundary_nodes = 2 * num_real_nodes
        else:
            num_boundary_nodes = 2 * num_real_nodes + 1
        
        # Step 5: Compute edge embeddings
        row, col = edge_index  # Separate the two nodes of each edge
        node_i = x[row]  # Node embeddings for source nodes
        node_j = x[col]  # Node embeddings for target nodes

        edge_embedding = torch.cat([node_i, node_j, edge_weights], dim=-1)  # [num_edges, 2 * hidden_dim + 1]

        # Step 6: Predict edge weight mean
        edge_weights_mean = self.edge_mlp(edge_embedding).squeeze(-1)  # [num_edges]
        
        return edge_index, edge_weights_mean,  num_real_nodes, num_boundary_nodes
        
    
    def filter_nodes_and_edges(self, x, stabilizer_type, edge_index, edge_weights):
        """
        Removes nodes that are not of type [1, 0] (X stabilizers) and updates
        the edge index by re-numbering the nodes starting from 0, to n_X_nodes

        Args:
            x    (torch.Tensor): Node features of shape [num_nodes, num_features].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            edge_weights (torch.Tensor): Edge weights of shape [num_edges, 1].

        Returns:
            new_x (torch.Tensor): Filtered node features.
            new_edge_index (torch.Tensor): Updated edge indices.
            new_edge_weights (torch.Tensor): Updated edge weights.
        """
        # Step 1: Identify valid nodes (type [1, 0])
        valid_nodes = (stabilizer_type[:, 0] == 1) & (stabilizer_type[:, 1] == 0)

        # Step 2: Create a mapping from old node indices to new indices
        new_node_ids = torch.full((x.shape[0],), -1, dtype=torch.long, device=self.device)  # -1 for removed nodes
        new_node_ids[valid_nodes] = torch.arange(valid_nodes.sum(), device=self.device)  # Renumber valid nodes
        # Step 3: Filter node features
        new_x = x[valid_nodes]  # Keep only valid nodes

        # Step 4: Filter edges where **both** nodes are valid
        src, tgt = edge_index
        valid_edges = valid_nodes[src] & valid_nodes[tgt]  # Ensure both endpoints are valid

        # Step 5: Update edge index with new node numbers
        new_edge_index = new_node_ids[edge_index[:, valid_edges]]  # Apply remapping
        new_edge_weights = edge_weights[valid_edges]  # Keep corresponding edge weights

        return new_x, new_edge_index, new_edge_weights

    
    def remove_duplicates(self, edge_index, edge_weights):
        """
        Removes duplicate undirected edges by keeping only (u, v) where u < v.
        Args:
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            edge_weights (torch.Tensor): Edge weights of shape [num_edges, 1].
        Returns:
            filtered_edge_index (torch.Tensor): Filtered edge indices.
            filtered_edge_weights (torch.Tensor): Corresponding edge weights.
        """
        mask = edge_index[0] < edge_index[1]  # Keep only edges where source < target
        filtered_edge_index = edge_index[:, mask]  # Apply mask to edge_index
        filtered_edge_weights = edge_weights[mask]  # Apply mask to edge_weight

        return filtered_edge_index, filtered_edge_weights
    
    def add_boundary_edges(self, edge_index, edge_weights, x):
        """
        Adds two unique boundary nodes for each real node:
            - One connecting to the left boundary (edge weight 1)
            - One connecting to the right boundary (edge weight 0)
        Args:
            x (torch.Tensor): Node features [num_nodes, feat_dim].
            edge_index (torch.Tensor): Edge indices [2, num_edges].
            edge_weights (torch.Tensor): Edge weights [num_edges, 1].
        Returns:
            edge_index (torch.Tensor): Updated edge indices with boundary edges.
            edge_weights (torch.Tensor): Updated edge weights.
            x (torch.Tensor): Updated node features including boundary nodes.
        Note: The virtual nodes on the western boundary have indices n, ... 2*n - 1
        """
        num_real_nodes = x.shape[0]

        # Create unique boundary node indices
        left_boundary_nodes = torch.arange(num_real_nodes, num_real_nodes * 2, device=self.device)  # Left boundary
        right_boundary_nodes = torch.arange(num_real_nodes * 2, num_real_nodes * 3, device=self.device)  # Right boundary

        # Create edges: (real_node -> left_boundary), (real_node -> right_boundary)
        left_edges = torch.stack([torch.arange(num_real_nodes, device=self.device), left_boundary_nodes], dim=0)
        right_edges = torch.stack([torch.arange(num_real_nodes, device=self.device), right_boundary_nodes], dim=0)

        # Assign weights: left boundary edge = 1, right boundary edge = -1
        left_weights = torch.ones((num_real_nodes, 1), device=self.device)  # Weight 1 for left boundary edges
        right_weights = -torch.ones((num_real_nodes, 1), device=self.device)  # Weight -1 for right boundary edges

        # Extend node features for boundary nodes (copy features from corresponding real nodes)
        x_boundary = x  # Duplicate real node features for boundary nodes
        x = torch.cat([x, x_boundary, x_boundary], dim=0)  # Expand node embeddings

        # Concatenate new boundary edges and weights
        edge_index = torch.cat([edge_index, left_edges, right_edges], dim=1)
        edge_weights = torch.cat([edge_weights, left_weights, right_weights], dim=0)

        return edge_index, edge_weights, x, num_real_nodes
    

def sample_weights_get_log_probs(edge_weights_mean, num_draws_per_sample, stddev, device):
    '''
    Compute the log-probabilities of the sampled edge weights.
    This is based on the Gaussian distribution with the predicted mean and stddev.
    Args:
        means: The predicted means from the GNN.
        stdev: the standard deviation of the policy
    Returns:
        sampled_edge_weights: The sampled edge weights.
        log_probs: The log-probabilities of the sampled edge weights
    '''
    num_edges = edge_weights_mean.shape[0]
    # Expand edge weights mean to match the number of draws per sample
    edge_weights_mean = edge_weights_mean.repeat(num_draws_per_sample, 1)
    # Sample from the Gaussian distribution (mean, stddev)
    with torch.no_grad():
        epsilon = torch.randn((num_draws_per_sample, num_edges), device=device)  # Standard normal noise
        sampled_edge_weights = edge_weights_mean + stddev * epsilon  # Sampled edge weights
    # Compute log-probabilities for the REINFORCE update
    # Log probability of the sampled value under the Gaussian distribution
    log_probs = - (sampled_edge_weights - edge_weights_mean)**2 / (2 * stddev**2)
    # Sum over all edge weights (shape (num_draws_per_sample,))
    log_probs = torch.sum(log_probs, dim=1)  

    return sampled_edge_weights, log_probs