from src.rotated_surface_code import RotatedCode
from src.graph_representation import get_syndrome_graph
from src.mwpm_prediction import compute_mwpm_reward
from src.gnn_model import EdgeWeightGNN, sample_weights_get_log_probs
from src.utils import test_model, save_checkpoint
import torch
import argparse
import numpy as np
import os
import time
from datetime import datetime
# python -m cProfile -o timing_code_cap.prof train_edge_predictor.py
# snakeviz timing_code_cap.prof
import wandb
os.environ["WANDB_SILENT"] = "True"

def main():
    time_start = time.perf_counter()
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Train the neural network with specified parameters")
    parser.add_argument('--d', type=int, required=True, help='The value of d')

    # Parse the arguments
    args = parser.parse_args()

    # Access the value of d
    d = args.d

    # Get the SLURM job ID from the environment
    job_id = os.environ.get('SLURM_JOB_ID', 'unknown')  # Default to 'unknown' if not running in SLURM


    p = 0.05
    code = RotatedCode(d)
    print(f'Training d = {d}.')
    num_samples_per_epoch = int(1e4)
    num_draws_per_sample = int(2e2)
    tot_num_samples = 0
    stddev = torch.tensor(0.05, dtype=torch.float32)
    lr = 1e-4
    num_epochs = 500

    hidden_channels_GCN = [32, 64, 128, 256]
    hidden_channels_MLP = [512, 256, 128, 64, 32]

    model = EdgeWeightGNN(hidden_channels_GCN = hidden_channels_GCN, hidden_channels_MLP = hidden_channels_MLP)
    device = torch.device('cpu')
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Check for checkpoint and load if available
    load_checkpoint_path = 'saved_models_code_capacity/d_11_250401_151300_6992492.pt'  # path of existing checkpoint
    current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    name = "d_" + str(d) + "_" + current_datetime
    save_checkpoint_path = f'saved_models_code_capacity/{name}_resume.pt'
    
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

    wandb.init(project="neural_matching_code_capacity", name = name, config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "num_samples_per_epoch": num_samples_per_epoch,
        "num_draws_per_sample": num_draws_per_sample,
        "stddev": stddev.item(),
        "d": d,
        "p": p,
        "hidden_channels_GCN": hidden_channels_GCN, 
        "hidden_channels_MLP": hidden_channels_MLP})

    for epoch in range(start_epoch, num_epochs):
        epoch_time_start = time.perf_counter()
        epoch_log_loss = 0
        epoch_reward = 0
        optimizer.zero_grad()

        # initiate the dataset:
        graph_list = []
        while len(graph_list) < num_samples_per_epoch:
            graph = get_syndrome_graph(code, p)
            if not graph == None:
                graph_list.append(graph)

        train_acc = 0
        for i in range(num_samples_per_epoch):  # Draw multiple samples per epoch
            data = graph_list[i]
            all_log_probs = []
            all_rewards = []
            # Forward pass: Get sampled edge weights and their log-probabilities
            edge_index, edge_weights_mean, num_real_nodes, num_boundary_nodes = \
                model(data.x, data.edge_index, data.edge_attr)
            train_reward = compute_mwpm_reward(edge_index, edge_weights_mean, num_real_nodes,num_boundary_nodes, data.y)
            train_acc += (train_reward + 1) / (2 * num_samples_per_epoch)
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
            # subtract the baseline:
            baseline = all_rewards.mean()
            # The loss per draw and per edge is the log-probability times the reward
            log_loss_per_draw = all_log_probs * (all_rewards - baseline) # Shape: (num_draws_per_sample, )

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
        })
        # Save the checkpoint after the current epoch
        save_checkpoint(model, optimizer, epoch, epoch_reward, train_acc, epoch_log_loss, save_checkpoint_path)

    time_end = time.perf_counter()
    print(f'Total training time: {time_end - time_start:.2f}s')
if __name__ == "__main__":
    main()

