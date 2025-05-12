import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class EdgeWeightGNN_stim(nn.Module):
    '''
    A Graph Neural Network (GNN) for predicting edge weights based on node features and edge embeddings.
    This model uses graph convolution layers (GCNs) for node feature transformation and an MLP for edge embedding
    to predict the edge weights means of a Gaussian distribution.

    Attributes:
        graph_conv_layers (nn.ModuleList): A list of GraphConv layers applied sequentially to update node features.
        edge_mlp (nn.Sequential): A multi-layer perceptron (MLP) for processing edge embeddings to predict edge weights.
    Args:
        hidden_channels_GCN (list): The hidden units in each graph convolution layer.
        hidden_channels_MLP (list): The hidden units in each dense layer.
    '''
    def __init__(
        self,
        n_node_features = 5,
        hidden_channels_GCN=[32, 64, 128, 256],
        hidden_channels_MLP=[512, 256, 128, 64, 32]):
        # num_classes is 1 for each head
        super().__init__()
        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_conv_layers = nn.ModuleList(
            [
                GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ])
        # Edge embedding MLP:
        self.edge_embedding_mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels_GCN[-1] + 1, hidden_channels_MLP[0]),
            nn.ReLU())
        # Dense layers
        self.dense_layers = nn.ModuleList(
            [
                nn.Linear(in_channels, out_channels)
                for (in_channels, out_channels) in zip(hidden_channels_MLP[:-1], hidden_channels_MLP[1:])
            ])
        # output followed by a sigmoid activation:
        self.ouput = nn.Sequential(
            nn.Linear(hidden_channels_MLP[-1], 1),
            nn.Sigmoid())
 
    def forward(self, x, edge_index, edge_weights):
        """
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, node_feat_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            edge_weights (torch.Tensor): Edge features of shape [num_edges, 1].
        Returns:
            edge_weights (torch.Tensor): Predicted edge weights of shape [num_edges].
        """
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
        edge_weights_mean = self.edge_embedding_mlp(edge_embedding)
        for dense in self.dense_layers:
            edge_weights_mean = F.relu(dense(edge_weights_mean))
        # Step 7: Sigmoind activation to ensure edge weights are in [0, 1]
        edge_weights_mean = self.ouput(edge_weights_mean)
        edge_weights_mean = edge_weights_mean.squeeze(-1)  # [num_edges]
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
        new_node_ids = torch.full((x.shape[0],), -1, dtype=torch.long)  # -1 for removed nodes
        new_node_ids[valid_nodes] = torch.arange(valid_nodes.sum())  # Renumber valid nodes

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
        left_boundary_nodes = torch.arange(num_real_nodes, num_real_nodes * 2)  # Left boundary
        right_boundary_nodes = torch.arange(num_real_nodes * 2, num_real_nodes * 3)  # Right boundary

        # Create edges: (real_node -> left_boundary), (real_node -> right_boundary)
        left_edges = torch.stack([torch.arange(num_real_nodes), left_boundary_nodes], dim=0)
        right_edges = torch.stack([torch.arange(num_real_nodes), right_boundary_nodes], dim=0)

        # Assign weights: left boundary edge = 1, right boundary edge = -1
        left_weights = torch.ones((num_real_nodes, 1))  # Weight 1 for left boundary edges
        right_weights = -torch.ones((num_real_nodes, 1))  # Weight -1 for right boundary edges

        # Extend node features for boundary nodes (copy features from corresponding real nodes)
        x_boundary = x  # Duplicate real node features for boundary nodes
        x = torch.cat([x, x_boundary, x_boundary], dim=0)  # Expand node embeddings

        # Concatenate new boundary edges and weights
        edge_index = torch.cat([edge_index, left_edges, right_edges], dim=1)
        edge_weights = torch.cat([edge_weights, left_weights, right_weights], dim=0)

        return edge_index, edge_weights, x, num_real_nodes
    

class EdgeWeightGNN(nn.Module):
    '''
    A Graph Neural Network (GNN) for predicting edge weights based on node features and edge embeddings.
    This model uses graph convolution layers (GCNs) for node feature transformation and an MLP for edge embedding
    to predict the edge weights means of a Gaussian distribution.

    Attributes:
        graph_conv_layers (nn.ModuleList): A list of GraphConv layers applied sequentially to update node features.
        edge_mlp (nn.Sequential): A multi-layer perceptron (MLP) for processing edge embeddings to predict edge weights.
    Args:
        hidden_channels_GCN (list): The hidden units in each graph convolution layer.
        hidden_channels_MLP (list): The hidden units in each dense layer.
    '''
    def __init__(
        self,
        n_node_features = 4,
        hidden_channels_GCN=[32, 64, 128],
        hidden_channels_MLP=[256, 128, 64, 32]):
        # num_classes is 1 for each head
        super().__init__()
        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_conv_layers = nn.ModuleList(
            [
                GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ])
        
        # Dense layers: one for each of the three edge types: internal, left, right
        self.dense_layers_internal = nn.ModuleList([
            nn.Linear(2 * hidden_channels_GCN[-1], hidden_channels_MLP[0])] +
            [nn.Linear(in_c, out_c) for in_c, out_c in zip(hidden_channels_MLP[:-1], hidden_channels_MLP[1:])] + 
            [nn.Linear(hidden_channels_MLP[-1], 1)])
        
        self.dense_layers_left = nn.ModuleList([
            nn.Linear(2 * hidden_channels_GCN[-1], hidden_channels_MLP[0])] +
            [nn.Linear(in_c, out_c) for in_c, out_c in zip(hidden_channels_MLP[:-1], hidden_channels_MLP[1:])] + 
            [nn.Linear(hidden_channels_MLP[-1], 1)])
        
        self.dense_layers_right = nn.ModuleList([
            nn.Linear(2 * hidden_channels_GCN[-1], hidden_channels_MLP[0])] +
            [nn.Linear(in_c, out_c) for in_c, out_c in zip(hidden_channels_MLP[:-1], hidden_channels_MLP[1:])] + 
            [nn.Linear(hidden_channels_MLP[-1], 1)])

        # output followed by a sigmoid activation:
        self.ouput = nn.Sequential(
            nn.Linear(hidden_channels_MLP[-1], 1),
            nn.Sigmoid())
        
    def forward(self, x, edge_index, edge_weights):
        """
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, node_feat_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            edge_weights (torch.Tensor): Edge features of shape [num_edges, 1].
        Returns:
            edge_weights (torch.Tensor): Predicted edge weights of shape [num_edges].
        """
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

        edge_embedding = torch.cat([node_i, node_j], dim=-1)  # [num_edges, 2 * hidden_dim]

        # Step 6: Predict edge weight mean (Separate edge types before passing to MLP)
        real_node_cutoff = num_real_nodes  # since boundary nodes come after real ones
        row, col = edge_index
        is_internal = (row < real_node_cutoff) & (col < real_node_cutoff)
        is_left = (row < real_node_cutoff) & (col >= real_node_cutoff) & (col < 2 * real_node_cutoff)
        is_right = (row < real_node_cutoff) & (col >= 2 * real_node_cutoff)

        # Predict separately
        outputs = []
        for mask, layers in zip([is_internal, is_left, is_right],
                                [self.dense_layers_internal, self.dense_layers_left, self.dense_layers_right]):
            if mask.sum() > 0:
                e = edge_embedding[mask]
                for layer in layers[:-1]:
                    e = F.relu(layer(e))
        # Step 7: Sigmoind activation to ensure edge weights are in [0, 1]
                e = layers[-1](e)
                e = torch.sigmoid(e).squeeze(-1)
                outputs.append((mask, e))
    
        # Allocate full tensor
        edge_weights_mean = torch.zeros(edge_embedding.size(0), device=edge_embedding.device)
        for mask, out in outputs:
            edge_weights_mean[mask] = out
    
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
        new_node_ids = torch.full((x.shape[0],), -1, dtype=torch.long)  # -1 for removed nodes
        new_node_ids[valid_nodes] = torch.arange(valid_nodes.sum())  # Renumber valid nodes

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
        left_boundary_nodes = torch.arange(num_real_nodes, num_real_nodes * 2)  # Left boundary
        right_boundary_nodes = torch.arange(num_real_nodes * 2, num_real_nodes * 3)  # Right boundary

        # Create edges: (real_node -> left_boundary), (real_node -> right_boundary)
        left_edges = torch.stack([torch.arange(num_real_nodes), left_boundary_nodes], dim=0)
        right_edges = torch.stack([torch.arange(num_real_nodes), right_boundary_nodes], dim=0)

        # Assign weights: left boundary edge = 1, right boundary edge = -1
        left_weights = torch.ones((num_real_nodes, 1))  # Weight 1 for left boundary edges
        right_weights = -torch.ones((num_real_nodes, 1))  # Weight -1 for right boundary edges

        # Extend node features for boundary nodes (copy features from corresponding real nodes)
        x_boundary = x  # Duplicate real node features for boundary nodes
        x = torch.cat([x, x_boundary, x_boundary], dim=0)  # Expand node embeddings

        # Concatenate new boundary edges and weights
        edge_index = torch.cat([edge_index, left_edges, right_edges], dim=1)
        edge_weights = torch.cat([edge_weights, left_weights, right_weights], dim=0)

        return edge_index, edge_weights, x, num_real_nodes
    

def sample_weights_get_log_probs(edge_weights_mean, num_draws_per_sample, stddev):
    '''
    Compute the log-probabilities of the sampled edge weights.
    This is based on the Gaussian distribution with the predicted mean and stddev.
    Args:
        means: The predicted means from the GNN.
        stdev: the standard deviation of the policy
    Returns:
        sampled_edge_weights: The sampled edge weights. shape: (num_draws_per_sample, num_edges)
        log_probs: The log-probabilities of the sampled edge weights
    '''
    num_edges = edge_weights_mean.shape[0]
    # Expand edge weights mean to match the number of draws per sample
    edge_weights_mean = edge_weights_mean.repeat(num_draws_per_sample, 1)
    # Sample from the Gaussian distribution (mean, stddev)
    with torch.no_grad():
        epsilon = torch.randn((num_draws_per_sample, num_edges))  # Standard normal noise
        sampled_edge_weights = edge_weights_mean + stddev * epsilon  # Sampled edge weights
    # Compute log-probabilities for the REINFORCE update
    # Log probability of the sampled value under the Gaussian distribution
    log_probs = - (sampled_edge_weights - edge_weights_mean)**2 / (2 * stddev**2)
    
    # Sum over all edge weights (shape (num_draws_per_sample,))
    log_probs = torch.sum(log_probs, dim=1)

    return sampled_edge_weights, log_probs

class EdgeWeightGNN_batch(nn.Module):
    '''
    A Graph Neural Network (GNN) for predicting edge weights based on node features and edge embeddings.
    This model uses graph convolution layers (GCNs) for node feature transformation and an MLP for edge embedding
    to predict the edge weights means of a Gaussian distribution.

    Attributes:
        graph_conv_layers (nn.ModuleList): A list of GraphConv layers applied sequentially to update node features.
        edge_mlp (nn.Sequential): A multi-layer perceptron (MLP) for processing edge embeddings to predict edge weights.
    Args:
        hidden_channels_GCN (list): The hidden units in each graph convolution layer.
        hidden_channels_MLP (list): The hidden units in each dense layer.
    '''
    def __init__(
        self,
        n_node_features = 4,
        hidden_channels_GCN=[32, 64, 128],
        hidden_channels_MLP=[256, 128, 64, 32]):
        # num_classes is 1 for each head
        super().__init__()
        # GCN layers
        channels = [n_node_features] + hidden_channels_GCN
        self.graph_conv_layers = nn.ModuleList(
            [
                GraphConv(in_channels, out_channels)
                for (in_channels, out_channels) in zip(channels[:-1], channels[1:])
            ])
        
        # Dense layers: one for each of the three edge types: internal, left, right
        self.dense_layers_internal = nn.ModuleList([
            nn.Linear(2 * hidden_channels_GCN[-1], hidden_channels_MLP[0])] +
            [nn.Linear(in_c, out_c) for in_c, out_c in zip(hidden_channels_MLP[:-1], hidden_channels_MLP[1:])] + 
            [nn.Linear(hidden_channels_MLP[-1], 1)])
        
        self.dense_layers_left = nn.ModuleList([
            nn.Linear(2 * hidden_channels_GCN[-1], hidden_channels_MLP[0])] +
            [nn.Linear(in_c, out_c) for in_c, out_c in zip(hidden_channels_MLP[:-1], hidden_channels_MLP[1:])] + 
            [nn.Linear(hidden_channels_MLP[-1], 1)])
        
        self.dense_layers_right = nn.ModuleList([
            nn.Linear(2 * hidden_channels_GCN[-1], hidden_channels_MLP[0])] +
            [nn.Linear(in_c, out_c) for in_c, out_c in zip(hidden_channels_MLP[:-1], hidden_channels_MLP[1:])] + 
            [nn.Linear(hidden_channels_MLP[-1], 1)])

        # output followed by a sigmoid activation:
        self.ouput = nn.Sequential(
            nn.Linear(hidden_channels_MLP[-1], 1),
            nn.Sigmoid())
        
    def forward(self, x, edge_index, edge_weights, batch):
        """
        Args:
            x (torch.Tensor): Node features of shape [num_nodes, node_feat_dim].
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
            edge_weights (torch.Tensor): Edge features of shape [num_edges, 1].
        Returns:
            edge_weights (torch.Tensor): Predicted edge weights of shape [num_edges].
        """
        # store the stabilizer type of the nodes
        stabilizer_type = x[:, -2:]
        # Step 1: Apply GCN layers to update node embeddings
        for graph_conv in self.graph_conv_layers:
            x = F.relu(graph_conv(x, edge_index, edge_weights))
            
        # Step 2: Only keep X type stabilizers:
        x, edge_index, batch = self.filter_nodes_and_edges_batch(
                                        x, stabilizer_type, edge_index, batch)
        
        # Step 3: Remove duplicates (need only to process (source, target)):
        edge_index = self.remove_duplicates(edge_index)

        # Step 4: Add boundary edges connecting each node to two virtual nodes
        real_node_counts = torch.bincount(batch)

        edge_index, x, batch, graph_info = self.add_boundary_edges_batch(
            edge_index, x, batch, real_node_counts)
        
        # Step 5: Compute edge embeddings
        row, col = edge_index  # Separate the two nodes of each edge
        node_i = x[row]  # Node embeddings for source nodes
        node_j = x[col]  # Node embeddings for target nodes

        edge_embedding = torch.cat([node_i, node_j], dim=-1)  # [num_edges, 2 * hidden_dim]

        # Step 6: Predict edge weight mean (Separate edge types before passing to MLP)
        row, col = edge_index
        is_internal = torch.zeros(row.shape[0], dtype=torch.bool)
        is_left = torch.zeros(row.shape[0], dtype=torch.bool)
        is_right = torch.zeros(row.shape[0], dtype=torch.bool)

        for info in graph_info:
            r_start = info['real_start']
            l_start = info['left_start']
            r_boundary_start = info['right_start']
            num_real = info['num_real']

            r_end = r_start + num_real
            l_end = l_start + num_real
            r_boundary_end = r_boundary_start + num_real

            # Internal edges: both ends in real nodes
            mask_internal = (row >= r_start) & (row < r_end) & \
                            (col >= r_start) & (col < r_end)

            # Left boundary edges: real → left
            mask_left = (row >= r_start) & (row < r_end) & \
                        (col >= l_start) & (col < l_end)

            # Right boundary edges: real → right
            mask_right = (row >= r_start) & (row < r_end) & \
                         (col >= r_boundary_start) & (col < r_boundary_end)

            is_internal |= mask_internal
            is_left |= mask_left
            is_right |= mask_right


        # Predict separately
        outputs = []
        for mask, layers in zip([is_internal, is_left, is_right],
                                [self.dense_layers_internal, self.dense_layers_left, self.dense_layers_right]):
            if mask.sum() > 0:
                e = edge_embedding[mask]
                for layer in layers[:-1]:
                    e = F.relu(layer(e))
        # Step 7: Sigmoind activation to ensure edge weights are in [0, 1]
                e = layers[-1](e)
                e = torch.sigmoid(e).squeeze(-1)
                outputs.append((mask, e))
    
        # Allocate full tensor
        edge_weights_mean = torch.zeros(edge_embedding.size(0))
        for mask, out in outputs:
            edge_weights_mean[mask] = out
    
        return edge_index, edge_weights_mean, batch, graph_info

        
    
    def filter_nodes_and_edges_batch(self, x, stabilizer_type, edge_index, batch):
        """
        Filters out non-X stabilizer nodes and updates edge_index and batch accordingly.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, num_features].
            stabilizer_type (torch.Tensor): Last two features of x, used to identify X stabilizers.
            edge_index (torch.Tensor): [2, num_edges] edge index tensor.
            batch (torch.Tensor): [num_nodes] batch tensor mapping nodes to graph ids.

        Returns:
            new_x (torch.Tensor): Filtered node features.
            new_edge_index (torch.Tensor): Updated edge indices.
            new_batch (torch.Tensor): Updated batch tensor.
        """
        # Identify valid nodes (X stabilizers: type [1, 0])
        valid_nodes = (stabilizer_type[:, 0] == 1) & (stabilizer_type[:, 1] == 0)

        # Create mapping from old node indices to new
        new_node_ids = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        new_node_ids[valid_nodes] = torch.arange(valid_nodes.sum(), device=x.device)

        # Filter node features and batch
        new_x = x[valid_nodes]
        new_batch = batch[valid_nodes]

        # Filter edges where both endpoints are valid
        src, tgt = edge_index
        valid_edges = valid_nodes[src] & valid_nodes[tgt]

        # Update edge_index
        new_edge_index = torch.stack([
            new_node_ids[src[valid_edges]],
            new_node_ids[tgt[valid_edges]]], dim=0)

        return new_x, new_edge_index, new_batch


    def remove_duplicates(self, edge_index):
        """
        Removes duplicate undirected edges by keeping only (u, v) where u < v.
        Args:
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
        Returns:
            filtered_edge_index (torch.Tensor): Filtered edge indices.
        """
        mask = edge_index[0] < edge_index[1]  # Keep only edges where source < target
        filtered_edge_index = edge_index[:, mask]  # Apply mask to edge_index

        return filtered_edge_index
    
    def add_boundary_edges_batch(self, edge_index, x, batch, real_node_counts):
        """
        Adds boundary nodes and edges for each graph in the batch.

        Args:
            edge_index (torch.Tensor): [2, num_edges]
            x (torch.Tensor): Node features [num_nodes, feat_dim]
            batch (torch.Tensor): [num_nodes], graph assignment for each node
            real_node_counts (torch.Tensor): [num_graphs], number of real nodes per graph

        Returns:
            edge_index (torch.Tensor): Updated with boundary edges
            x (torch.Tensor): Extended with boundary node embeddings
            graph_info (List[Dict]): Per-graph node indexing information
        """
        device = x.device
        total_real_nodes = x.shape[0]
        num_graphs = real_node_counts.shape[0]

        edge_index_list = [edge_index]
        x_list = [x]
        batch_list = [batch]  # initialize with existing batch entries
        graph_info = []

        real_node_offsets = torch.cat([
            torch.tensor([0], device=real_node_counts.device),
            torch.cumsum(real_node_counts[:-1], dim=0)
        ])

        boundary_node_offset = total_real_nodes  # start of boundary nodes

        for g in range(num_graphs):
            num_real = real_node_counts[g].item()
            real_start = real_node_offsets[g].item()
            left_start = boundary_node_offset
            right_start = left_start + num_real

            # Real node indices for this graph
            real_ids = torch.arange(real_start, real_start + num_real, device=device)

            # Boundary node indices
            left_ids = torch.arange(left_start, left_start + num_real, device=device)
            right_ids = torch.arange(right_start, right_start + num_real, device=device)

            # Append boundary node graph IDs to batch
            boundary_batch = torch.full((2 * num_real,), g, dtype=torch.long, device=device)
            batch_list.append(boundary_batch)

            # Boundary node features
            real_x = x[real_ids]
            x_left = real_x.clone()
            x_right = -real_x.clone()  # NEGATED for right boundary

            x_list.append(x_left)
            x_list.append(x_right)

            # Edges: real → left, real → right
            left_edges = torch.stack([real_ids, left_ids], dim=0)
            right_edges = torch.stack([real_ids, right_ids], dim=0)

            edge_index_list.extend([left_edges, right_edges])

            # Advance boundary node index
            boundary_node_offset = right_start + num_real

            num_boundary_nodes = 2 * num_real if num_real % 2 == 0 else 2 * num_real + 1

            graph_info.append({
                'real_start': real_start,
                'left_start': left_start,
                'right_start': right_start,
                'num_real': num_real,
                'num_boundary': num_boundary_nodes
            })

        edge_index = torch.cat(edge_index_list, dim=1)
        x = torch.cat(x_list, dim=0)
        batch = torch.cat(batch_list, dim=0)

        return edge_index, x, batch, graph_info

