from src.get_stim_syndromes import initialize_simulations
from src.mwpm_prediction import compute_mwpm_reward
from src.gnn_model import EdgeWeightGNN_stim, sample_weights_get_log_probs
from src.utils import test_model, save_checkpoint, get_acc_from_csv
from src.graph_representation_stim import get_syndrome_graph
import torch
import numpy as np

def main():
    p = 0.001
    d = 9
    d_t = d
    acc_mwpm = get_acc_from_csv('/Users/xlmori/Desktop/neural_matching/mwpm_stim_p_1e-3_5e-3_results.csv', d, d_t, p)
    compiled_sampler, syndrome_mask, detector_coordinates = initialize_simulations(d, d_t, p)

    print(f'Training d = {d}, d_t = {d_t}.')
    num_samples_per_epoch = int(1e3)
    num_draws_per_sample = int(1e2)
    tot_num_samples = 0
    test_set_size = int(1e4)
    stddev = torch.tensor(0.1, dtype=torch.float32)
    lr = 1e-3
    num_epochs = 500

    # initiate the test dataset:
    test_set = []
    test_n_trivials = 0
    for _ in range(test_set_size):
        graph = get_syndrome_graph(compiled_sampler, syndrome_mask, detector_coordinates)
        if not graph == None:
           test_set.append(graph)
        else: 
            test_n_trivials += 1
    n_nontrivial_test_samples = len(test_set)
    n_trivial_test_samples = test_set_size - n_nontrivial_test_samples

    model = EdgeWeightGNN_stim()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Check for checkpoint and load if available
    # generate a unique name to not overwrite other models
    name = "d_" + str(d) + "_d_t_" + str(d_t) + "_p_0p" + f"{p:.3f}".split(".")[1]
    checkpoint_path = 'saved_models/stim_gcn_32_64_128_mlp_128_64_32_memory_x/' + name + '.pt'
    start_epoch = 0
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        start_epoch = checkpoint['epoch']  # Get the epoch from checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load optimizer state
        print(f"Checkpoint loaded, continuing from epoch {start_epoch}.")
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
    num_epochs = start_epoch + num_epochs
    # Learning rate:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for epoch in range(start_epoch, num_epochs):
        epoch_log_loss = 0
        epoch_reward = 0
        optimizer.zero_grad()

        # initiate the dataset:
        graph_list = []
        while len(graph_list) < num_samples_per_epoch:
            graph = get_syndrome_graph(compiled_sampler, syndrome_mask, detector_coordinates)
            if not graph == None:
                graph_list.append(graph)

        for i in range(num_samples_per_epoch):  # Draw multiple samples per epoch
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
                reward = compute_mwpm_reward(edge_index, edge_weights_j, num_real_nodes,num_boundary_nodes, data.y)
                # Store log-probabilities and rewards
                all_log_probs.append(log_probs[j])
                all_rewards.append(reward)
            # Stack log-probs and rewards for averaging
            all_log_probs = torch.stack(all_log_probs)  # Shape: (num_draws_per_sample,)
            all_rewards = torch.tensor(all_rewards, dtype=torch.float32) # Shape: (num_draws_per_sample,)
            # The loss per draw and per edge is the log-probability times the reward
            log_loss_per_draw = all_log_probs * all_rewards  # Shape: (num_draws_per_sample, )

            # Compute the REINFORCE loss for each edge
            # maximize the reward, so minimize - reward
            log_loss_per_sample = -torch.mean(log_loss_per_draw)  # Shape: (1,)
            mean_reward_per_sample = torch.mean(all_rewards)
            log_loss_per_sample.backward()  # Accumulate gradients
            epoch_log_loss += log_loss_per_sample.item()
            epoch_reward += mean_reward_per_sample.item()

        optimizer.step()  # Perform a single optimization step after accumulating gradients
        epoch_log_loss /= num_samples_per_epoch
        epoch_reward /= num_samples_per_epoch
        tot_num_samples += num_samples_per_epoch

        train_acc = test_model(model, num_samples_per_epoch, graph_list)

        test_acc_nontrivial = test_model(model, n_nontrivial_test_samples, test_set)
        num_corr_nontrivial = test_acc_nontrivial * n_nontrivial_test_samples
        test_acc = (num_corr_nontrivial + n_trivial_test_samples) / test_set_size
        # Print training progress
        if epoch % 1 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Log-Loss: {epoch_log_loss:.4f}, Mean Reward: {epoch_reward:.4f}, No. Samples: {tot_num_samples}, Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}, MWPM: {1 -acc_mwpm:.4f}, Diff: {test_acc - (1 - acc_mwpm):.4f}')

        # Save the checkpoint after the current epoch
        save_checkpoint(model, optimizer, epoch, epoch_reward, train_acc, test_acc, checkpoint_path)
if __name__ == "__main__":
    main()

