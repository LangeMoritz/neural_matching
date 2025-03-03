from rotated_surface_code import RotatedCode
from graph_representation import get_syndrome_graph
from mwpm_prediction import compute_mwpm_reward_parallel
from gnn_model import EdgeWeightGNN, sample_weights_get_log_probs
from utils import test_model
import torch
import numpy as np
from multiprocessing import Pool, cpu_count

# python -m cProfile -o output_mwpm_parallel.prof src/train_edge_predictor_mwpm_parallel.py
# snakeviz output_mwpm_parallel.prof

def main():
    p = 0.1
    d = 5
    code = RotatedCode(d)

    num_samples_per_epoch = 100
    num_draws_per_sample = 10
    tot_num_samples = 0
    stddev = torch.tensor(0.1, dtype=torch.float32)


    model = EdgeWeightGNN()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 2

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_reward = 0
        optimizer.zero_grad()

        # initiate the dataset:
        graph_list = []
        while len(graph_list) < num_samples_per_epoch:
            graph = get_syndrome_graph(code, p)
            if not graph == None:
                graph_list.append(graph)

        for i in range(num_samples_per_epoch):  # Draw multiple samples per epoch
            data = graph_list[i]
            all_log_probs = []
            # Forward pass: Get sampled edge weights and their log-probabilities
            edge_index, edge_weights_mean, num_real_nodes, num_boundary_nodes = \
                model(data.x, data.edge_index, data.edge_attr)
            sampled_edge_weights, log_probs = sample_weights_get_log_probs(edge_weights_mean, num_draws_per_sample, stddev)

            edge_index = edge_index.detach().numpy()
            sampled_edge_weights = sampled_edge_weights.detach().numpy()

            # create list of arguments for the multiprocesses 
            repeated_arguments = []
            for j in range(num_draws_per_sample):
                y_numpy = data.y.detach().numpy()
                repeated_arguments.append((edge_index, sampled_edge_weights[j, :], num_real_nodes, num_boundary_nodes, y_numpy))
                all_log_probs.append(log_probs[j])
            # compute rewards for all draws in parallel:
            with Pool(processes = (cpu_count() - 1)) as pool:
                all_rewards = pool.starmap(compute_mwpm_reward_parallel, repeated_arguments)
            
            # Stack log-probs and rewards for averaging
            all_log_probs = torch.stack(all_log_probs)  # Shape: (num_draws_per_sample,)
            all_rewards = torch.tensor(all_rewards, dtype=torch.float32)            # Shape: (num_draws_per_sample,)
            # The loss per draw and per edge is the log-probability times the reward
            loss_per_draw = all_log_probs * all_rewards  # Shape: (num_draws_per_sample, )

            # Compute the REINFORCE loss for each edge
            # maximize the reward, so minimize - reward
            loss_per_sample = -torch.mean(loss_per_draw)  # Shape: (1,)
            mean_reward_per_sample = torch.mean(all_rewards)
            loss_per_sample.backward()  # Accumulate gradients
            epoch_loss += loss_per_sample.item()
            epoch_reward += mean_reward_per_sample.item()

        optimizer.step()  # Perform a single optimization step after accumulating gradients
        epoch_loss /= num_samples_per_epoch
        epoch_reward /= num_samples_per_epoch
        tot_num_samples += num_samples_per_epoch

        train_acc = test_model(model, num_samples_per_epoch, graph_list)
        # Print training progress
        if epoch % 1 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {epoch_loss:.4f}, Mean Reward: {epoch_reward:.4f}, No. Samples: {tot_num_samples}, Accuracy: {train_acc:.4f}')

if __name__ == "__main__":
    main()
