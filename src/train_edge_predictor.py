from rotated_surface_code import RotatedCode
from graph_representation import get_syndrome_graph
from mwpm_prediction import compute_mwpm_reward
from gnn_model import EdgeWeightGNN, sample_weights_get_log_probs
import torch
import numpy as np
from multiprocessing import Pool, cpu_count

# python -m cProfile -o output.prof src/train_edge_predictor.py
# snakeviz output.prof

def main():
    p = 0.1
    d = 5
    code = RotatedCode(d)

    num_samples = 100
    num_draws_per_sample = 100
    stddev = torch.tensor(0.1, dtype=torch.float32)

    graph_list = []
    while len(graph_list) < num_samples:
        graph = get_syndrome_graph(code, p)
        if not graph == None:
            graph_list.append(graph)

    model = EdgeWeightGNN()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10000

    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_reward = []
        for i in range(num_samples):  # Draw multiple samples per epoch
            data = graph_list[i]
            all_log_probs = []
            all_rewards = []
            # Forward pass: Get sampled edge weights and their log-probabilities
            edge_index, edge_weights_mean, num_real_nodes, num_boundary_nodes = \
                model(data.x, data.edge_index, data.edge_attr)
            sampled_edge_weights, log_probs = sample_weights_get_log_probs(edge_weights_mean, num_draws_per_sample, stddev)
            # sampled_edge_weights = torch.sigmoid(sampled_edge_weights)
            for j in range(num_draws_per_sample):
                edge_weights_j = sampled_edge_weights[j, :]
                reward, matching = compute_mwpm_reward(edge_index, edge_weights_j, num_real_nodes,num_boundary_nodes, data.y)
                # Store log-probabilities and rewards
                all_log_probs.append(log_probs[j])
                all_rewards.append(reward)
            # Stack log-probs and rewards for averaging
            all_log_probs = torch.stack(all_log_probs)  # Shape: (num_draws_per_sample,)
            all_rewards = torch.tensor(all_rewards, dtype=torch.float32)            # Shape: (num_draws_per_sample,)
            # The loss per draw and per edge is the log-probability times the reward
            loss_per_draw = all_log_probs * all_rewards  # Shape: (num_draws_per_sample, )

            # Compute the REINFORCE loss for each edge
            loss_per_sample = -torch.mean(loss_per_draw)  # Shape: (1,)
            mean_reward_per_sample = torch.mean(all_rewards)
            optimizer.zero_grad()
            loss_per_sample.backward()  # Provide gradient for each sample
            optimizer.step()

            # mean over all samples
            epoch_reward.append(mean_reward_per_sample.item())
            epoch_loss.append(loss_per_sample.mean().item())
        epoch_loss = torch.tensor(epoch_loss).mean()
        epoch_reward = torch.tensor(epoch_reward).mean()
        # Print training progress
        if epoch % 500 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Mean Reward: {epoch_reward:.4f}')

if __name__ == "__main__":
    main()

