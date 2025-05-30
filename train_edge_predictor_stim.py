from src.get_stim_syndromes import initialize_simulations
from src.mwpm_prediction import compute_mwpm_reward
from src.gnn_model import EdgeWeightGNN_stim, sample_weights_get_log_probs
from src.utils import test_model, save_checkpoint, get_acc_from_csv
from src.graph_representation_stim import get_syndrome_graph
import torch
import time
import numpy as np
from datetime import datetime
import os
import argparse
# python -m cProfile -o timing_circuit_level.prof train_edge_predictor.py
# snakeviz timing_circuit_level.prof
import wandb
os.environ["WANDB_SILENT"] = "True"

def main():
    time_start = time.perf_counter()
    time_sampling = 0
    time_mwpm = 0
    p = 0.001
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Train the neural network with specified parameters")
    parser.add_argument('--d', type=int, required=True, help='The value of d')

    # Parse the arguments
    args = parser.parse_args()

    # Access the value of d
    d = args.d
    d_t = d
    acc_mwpm = 1.0 #get_acc_from_csv('/Users/xlmori/Desktop/neural_matching/mwpm_stim_p_1e-3_5e-3_results.csv', d, d_t, p)
    compiled_sampler, syndrome_mask, detector_coordinates = initialize_simulations(d, d_t, p)

    print(f'Training d = {d}, d_t = {d_t}.')
    num_samples_per_epoch = int(1e4)
    num_draws_per_sample = int(1e1)
    tot_num_samples = 0
    stddev = torch.tensor(0.1, dtype=torch.float32, device = torch.device('cpu'))
    lr = 1e-5
    num_epochs = 500

    hidden_channels_GCN = [32, 64, 128, 256]
    hidden_channels_MLP = [512, 256, 128, 64, 32]

    model = EdgeWeightGNN_stim(hidden_channels_GCN = hidden_channels_GCN, hidden_channels_MLP = hidden_channels_MLP)
    device = torch.device('cpu')
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Check for checkpoint and load if available
    load_checkpoint_path = 'saved_models/d_9_d_t_9_250330_210913.pt'  # path of existing checkpoint
    current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    name = "d_" + str(d) + "_d_t_" + str(d_t) + "_" + current_datetime
    save_checkpoint_path = f'saved_models/{name}_baseline_resume.pt'

    start_epoch = 0
    try:
        checkpoint = torch.load(load_checkpoint_path, weights_only=True)
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
    wandb.init(project="neural_matching", name = name, config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "num_samples_per_epoch": num_samples_per_epoch,
        "num_draws_per_sample": num_draws_per_sample,
        "stddev": stddev.item(),
        "d": d,
        "d_t": d_t,
        "p": p,
        "hidden_channels_GCN": hidden_channels_GCN, 
        "hidden_channels_MLP": hidden_channels_MLP
    })

    baseline = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_time_start = time.perf_counter()
        epoch_log_loss = 0
        epoch_reward = 0
        optimizer.zero_grad()

        # initiate the dataset:
        time_sampling_start = time.perf_counter()
        graph_list = []
        while len(graph_list) < num_samples_per_epoch:
            graph = get_syndrome_graph(compiled_sampler, syndrome_mask, detector_coordinates)
            if not graph == None:
                graph_list.append(graph)
        time_sampling += (time.perf_counter() - time_sampling_start)
        train_acc = 0
        for i in range(num_samples_per_epoch):  # Draw multiple samples per epoch
            data = graph_list[i]
            all_log_probs = []
            all_rewards = []
            # Forward pass: Get sampled edge weights and their log-probabilities
            edge_index, edge_weights_mean, num_real_nodes, num_boundary_nodes = \
                model(data.x, data.edge_index, data.edge_attr)

            # get accuracy of the means:
            time_mwpm_start = time.perf_counter()
            train_reward = compute_mwpm_reward(edge_index, edge_weights_mean, num_real_nodes,num_boundary_nodes, data.y)
            train_acc += (train_reward + 1) / (2 * num_samples_per_epoch)
            time_mwpm += (time.perf_counter() - time_mwpm_start)

            sampled_edge_weights, log_probs = sample_weights_get_log_probs(edge_weights_mean, num_draws_per_sample, stddev)

            time_mwpm_start = time.perf_counter()
            for j in range(num_draws_per_sample):
                edge_weights_j = sampled_edge_weights[j, :]
                reward = compute_mwpm_reward(edge_index, edge_weights_j, num_real_nodes,num_boundary_nodes, data.y)
                # Store log-probabilities and rewards
                all_log_probs.append(log_probs[j])
                all_rewards.append(reward)
            time_mwpm += (time.perf_counter() - time_mwpm_start)

            # Stack log-probs and rewards for averaging
            all_log_probs = torch.stack(all_log_probs)  # Shape: (num_draws_per_sample,)
            all_rewards = torch.tensor(all_rewards, dtype=torch.float32, device = torch.device('cpu')) # Shape: (num_draws_per_sample,)
            # The loss per draw and per edge is the log-probability times the reward - baseline
            log_loss_per_draw = all_log_probs * (all_rewards - baseline) # Shape: (num_draws_per_sample, )

            # Compute the REINFORCE loss for each edge
            # maximize the reward, so minimize - reward
            log_loss_per_sample = -torch.mean(log_loss_per_draw)  # Shape: (1,)
            mean_reward_per_sample = torch.mean(all_rewards)
            log_loss_per_sample.backward()  # Accumulate gradients
            epoch_log_loss += log_loss_per_sample.item()
            epoch_reward += mean_reward_per_sample.item()

        # Normalize the accumulated gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad /= num_samples_per_epoch
        optimizer.step()  # Perform a single optimization step after accumulating gradients
        epoch_log_loss /= num_samples_per_epoch
        epoch_reward /= num_samples_per_epoch
        baseline = (baseline + epoch_reward) / 2
        tot_num_samples += num_samples_per_epoch

        epoch_time = time.perf_counter() - epoch_time_start
        # Print training progress
        if epoch % 1 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Log-Loss: {epoch_log_loss:.4f}, Mean Reward: {epoch_reward:.4f}, No. Samples: {tot_num_samples}, Accuracy: {train_acc:.4f}, Time: {epoch_time:.2f} seconds')
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "log_loss": epoch_log_loss,
            "mean_reward": epoch_reward,
            "time": epoch_time,
            "accuracy": train_acc,
        })
        # Save the checkpoint after the current epoch
        save_checkpoint(model, optimizer, epoch, epoch_reward, train_acc, epoch_log_loss, save_checkpoint_path)
        
    time_end = time.perf_counter()
    print(f'Total training time: {time_end - time_start:.2f}s, thereof sampling: {time_sampling:.2f}s and MWPM: {time_mwpm:.2f}s')
if __name__ == "__main__":
    main()
