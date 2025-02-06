from src.rotated_surface_code import RotatedCode
from src.graph_representation import get_torch_graph
from src.mwpm_prediction import compute_mwpm_reward
import torch

def train_gnn_with_rl_multi_sample(model, graph_list, optimizer, num_epochs, num_samples, device):
    """
    Train the GNN with reinforcement learning (REINFORCE framework) using multiple samples
    to reduce the variance of the gradient estimate.

    Args:
        model: The GNN model (GCNEdgeWeightModel).
        data: The graph data (PyTorch Geometric Data object).
        optimizer: Optimizer (e.g., Adam) for the model.
        num_epochs: Number of training epochs.
        num_samples: Number of samples to draw per epoch.
        device: Device for computation (e.g., 'cuda' or 'cpu').
    """
    model.to(device)
    data.to(device)

    for epoch in range(num_epochs):
        for i in range(num_samples):  # Draw multiple samples per epoch
            data = graph_list[i]
            all_log_probs = []
            all_rewards = []
            # Forward pass: Get sampled edge weights and their log-probabilities
            edge_index, edge_weights_mean, num_real_nodes = model(data.x,
                                                                  data.edge_index,
                                                                  data.edge_attr)
            sampled_edge_weights, log_probs = sample_edge_weights(edge_weights_mean, num_draws_per_sample, stddev)
            # print(sampled_edge_weights)
            # Compute the reward for the current sample (interact with MWPM)
            for j in range(num_draws_per_sample):
                edge_weights_j = sampled_edge_weights[j, :]
                reward = compute_mwpm_reward(edge_index, edge_weights_j, num_real_nodes, data.y)
                # Store log-probabilities and rewards
                all_log_probs.append(log_probs[j, :])
                all_rewards.append(reward)
            # Stack log-probs and rewards for averaging
            all_log_probs = torch.stack(all_log_probs, dim=0)  # Shape: (num_draws_per_sample, num_edges)
            all_rewards = torch.tensor(all_rewards, dtype=torch.float32)            # Shape: (num_draws_per_sample,)
            mean_reward = torch.mean(all_rewards)
            # The loss per draw and per edge is the log-probability times the reward
            loss_per_draw = all_log_probs * all_rewards[:, None]# Shape: (num_draws_per_sample, num_edges)

            # Compute the REINFORCE loss
            loss = - torch.mean(loss_per_draw) #, dim=0)           # Shape: (1, num_edges)
            # mean_log_probs = torch.mean(all_log_probs, dim=0)  # Mean log-probs across draws
            # mean_rewards = torch.mean(all_rewards, dim=0)      # Mean rewards across draws
            # loss = torch.mean(mean_log_probs * mean_rewards)  # mean across edges from one sample
            # Backpropagate and update the model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Print training progress
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, Mean Reward: {mean_reward.item():.4f}')