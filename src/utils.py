"""Utility functions and layers for the FEC package."""

import numpy as np
import yaml
import torch
from src.mwpm_prediction import compute_mwpm_reward

# Save the entire training history along with model and optimizer states
def save_checkpoint(model, optimizer, epoch, epoch_reward, train_acc, test_acc, checkpoint_path):
    try:
        # Load existing checkpoint if it exists
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        history = checkpoint.get('history', [])
    except (FileNotFoundError, RuntimeError):
        history = []

    # Append the new epoch data
    history.append({'epoch': epoch + 1, 'epoch_reward': epoch_reward, 'train_acc': train_acc, 'test_acc': test_acc})

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