import numpy as np
import yaml
import torch
from src.mwpm_prediction import compute_mwpm_reward
import pandas as pd

def sample_weights_get_log_probs_batch(edge_weights_mean, edge_index, batch, num_draws_per_sample, num_samples_per_epoch, stddev):
    '''
    Sample edge weights and compute per-graph log-probabilities for REINFORCE, using a loop.

    Args:
        edge_weights_mean (torch.Tensor): shape [num_edges], predicted means per edge.
        edge_index (torch.Tensor): shape [2, num_edges], edge list.
        batch (torch.Tensor): shape [num_nodes], updated batch tensor after adding boundary nodes.
        num_draws_per_sample (int): Number of samples per graph.
        num_samples_per_epoch (int): Number of samples per epoch.
        stddev (float): Standard deviation of the Gaussian sampling policy.

    Returns:
        sampled_edge_weights (torch.Tensor): shape [num_draws_per_sample, num_edges]
        log_probs_per_graph (torch.Tensor): shape [num_draws_per_sample, num_graphs]
    '''
    num_edges = edge_weights_mean.shape[0]
    # Assign each edge to a graph using the graph ID of the source node
    edge_graph_indicator = batch[edge_index[0]]  # shape [num_edges]
    
    # Sample edge weights
    with torch.no_grad():
        epsilon = torch.randn((num_draws_per_sample, num_edges))
        edge_weights_mean_expanded = edge_weights_mean.unsqueeze(0).expand(num_draws_per_sample, -1)
        sampled_edge_weights = edge_weights_mean_expanded + stddev * epsilon

    # Compute log-probs per edge
    log_probs_per_edge = - (sampled_edge_weights - edge_weights_mean_expanded)**2 / (2 * stddev**2)

    # Sum log-probs per graph
    log_probs_per_graph = torch.zeros((num_draws_per_sample, num_samples_per_epoch))
    for g in range(num_samples_per_epoch):
        edge_mask = (edge_graph_indicator == g)
        log_probs_per_graph[:, g] = log_probs_per_edge[:, edge_mask].sum(dim=1)
    # have shape (num_draws_per_sample, num_all_edges) and (num_draws_per_sample, num_samples_per_epoch)
    return sampled_edge_weights, log_probs_per_graph


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


def get_acc_from_csv(file_path, d, d_t, p):
    """
    Reads the CSV file and returns the 'acc' value for the given d, d_t, and p.
    
    Args:
        file_path (str): Path to the CSV file.
        d (float): Value of d.
        d_t (float): Value of d_t.
        p (float): Value of p.
        
    Returns:
        float: The corresponding acc value, or None if not found.
    """
    df = pd.read_csv(file_path)
    
    # Ensure the numeric columns are treated as floats
    df[['d', 'd_t', 'p']] = df[['d', 'd_t', 'p']].astype(float)
    
    # Filter the dataframe for the given values
    match = df[(df['d'] == d) & (df['d_t'] == d_t) & (df['p'] == p)]
    
    if not match.empty:
        return match['acc'].values[0]  # Return the first match
    else:
        return None
# Save the entire training history along with model and optimizer states
def save_checkpoint(model, optimizer, epoch, epoch_reward, train_acc, epoch_log_loss, checkpoint_path):
    try:
        # Load existing checkpoint if it exists
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        history = checkpoint.get('history', [])
    except (FileNotFoundError, RuntimeError):
        history = []

    # Append the new epoch data
    history.append({'epoch': epoch + 1, 'epoch_reward': epoch_reward, 'train_acc': train_acc, 'epoch_log_loss': epoch_log_loss})

    # Save updated checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(checkpoint, checkpoint_path)


def parse_yaml(yaml_config):
    
    if yaml_config is not None:
        with open(yaml_config, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
    # default settings
    else:
        config = {}
        config["paths"] = {
            "root": "../",
            "save_dir": "../training_outputs",
            "model_name": "graph_decoder"
        }
        config["model_settings"] = {
            "hidden_channels_GCN": [32, 128, 256, 512, 512, 256, 256],
            "hidden_channels_MLP": [256, 128, 64],
            "num_classes": 12
        }
        config["graph_settings"] = {
            "code_size": 7,
            "error_rate": 0.001,
            "m_nearest_nodes": 5
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config["training_settings"] = {
            "seed": None,
            "dataset_size": 50000,
            "batch_size": 4096,
            "epochs": 5,
            "lr": 0.01,
            "device": device,
            "resume_training": False,
            "current_epoch": 0
        }
    
    # read settings into variables
    paths = config["paths"]
    model_settings = config["model_settings"]
    graph_settings = config["graph_settings"]
    training_settings = config["training_settings"]
    
    return paths, model_settings, graph_settings, training_settings

def test_model(model, num_samples, graph_list):
    '''
    Test the model by generating new samples and computing the average accuracy.
    Args:
        model: The trained GNN model.
        num_samples: The number of samples to generate.
    Returns:
        mean_accuracy: The average accuracy over the generated samples.
    '''
    mean_accuracy = 0
    with torch.no_grad():
        for i in range(num_samples):
            data = graph_list[i]
            edge_index, edge_weights_mean, num_real_nodes, num_boundary_nodes = \
                    model(data.x, data.edge_index, data.edge_attr)
            reward = compute_mwpm_reward(edge_index, edge_weights_mean, num_real_nodes,num_boundary_nodes, data.y)
            accuracy = (reward + 1) / 2

            mean_accuracy += accuracy
        mean_accuracy /= num_samples
    return mean_accuracy