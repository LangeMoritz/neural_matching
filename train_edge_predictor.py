from src.rotated_surface_code import RotatedCode
from src.graph_representation import get_syndrome_graph
from src.mwpm_prediction import compute_mwpm_reward, compute_mwpm_rewards_multiple_draws, compute_mwpm_rewards_batched
from src.gnn_model import EdgeWeightGNN_batch
from src.utils import save_checkpoint, sample_weights_get_log_probs_batch
import torch
import numpy as np
import os
import time
from datetime import datetime
from torch_geometric.data import Batch
import gc
import argparse
# python -m cProfile -o figures_and_outputs/multi_syndromes.prof train_edge_predictor.py
# snakeviz figures_and_outputs/multi_syndromes.prof
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
    # d = 3
    p = [0.01, 0.05, 0.1, 0.15]  # physical error rates
    code = RotatedCode(d)
    print(f'Training d = {d}.')
    num_samples_per_epoch = int(1e4)
    num_draws_per_sample = int(2e2)
    tot_num_samples = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")
    stddev = torch.tensor(0.05, dtype=torch.float32, device=device)
    lr = 1e-5
    num_epochs = 2000

    hidden_channels_GCN = [32, 64, 128, 256]
    hidden_channels_MLP = [512, 128, 64]

    model = EdgeWeightGNN_batch(hidden_channels_GCN = hidden_channels_GCN, hidden_channels_MLP =hidden_channels_MLP)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Check for checkpoint and load if available
    model_folder = 'saved_models_code_capacity/'
    loaded_model_name = 'd9_depol_4463699'
    load_checkpoint_path =  model_folder + loaded_model_name + '.pt'  # path of existing checkpoint
    # current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    save_checkpoint_name = 'd' + str(d) + '_depol_' + job_id + '.pt'
    save_checkpoint_name = loaded_model_name + '_resume_' + job_id + '.pt'
    save_checkpoint_path = model_folder + save_checkpoint_name
    
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

    wandb.init(project="neural_matching_code_capacity", name = save_checkpoint_name, config = {
        "learning_rate": lr,
        "epochs": num_epochs,
        "num_samples_per_epoch": num_samples_per_epoch,
        "num_draws_per_sample": num_draws_per_sample,
        "stddev": stddev.item(),
        "d": d,
        "p": p,
        "hidden_channels_GCN": hidden_channels_GCN, 
        "hidden_channels_MLP": hidden_channels_MLP})

    baseline = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_time_start = time.perf_counter()
        optimizer.zero_grad()
        # initiate the dataset:
        graph_list = []
        while len(graph_list) < num_samples_per_epoch:
            p_one_sample = np.random.choice(p)
            graph = get_syndrome_graph(code, p_one_sample)
            if not graph == None:
                graph_list.append(graph)
        graph_batch = Batch.from_data_list(graph_list).to(device)
        
        # Forward pass: Get edge weights and number of real and boundary nodes
        edge_index, edge_weights_mean, batch, graph_info = model(graph_batch.x,
                                                          graph_batch.edge_index,
                                                          graph_batch.edge_attr,
                                                          graph_batch.batch)
        # have shape (num_draws_per_sample, num_all_edges) and (num_draws_per_sample, num_samples_per_epoch)
        sampled_edge_weights, all_log_probs = sample_weights_get_log_probs_batch(
            edge_weights_mean,
            edge_index,
            batch,
            num_draws_per_sample,
            num_samples_per_epoch,
            stddev)
        # has shape: (num_draws_per_sample, num_samples_per_epoch)
        rewards = compute_mwpm_rewards_batched(edge_index,
                                     sampled_edge_weights,
                                     graph_info,
                                     graph_batch.y)
        # The loss per draw and per edge is the log-probability times the reward - baseline
        all_expected_reward_gradients = - all_log_probs * (rewards - baseline)
        epoch_reward = torch.mean(rewards).item()
        
        epoch_expected_reward_gradient = torch.mean(all_expected_reward_gradients)
        epoch_expected_reward_gradient.backward()
        
        optimizer.step()

        baseline = (baseline + epoch_reward) / 2
        tot_num_samples += num_samples_per_epoch

        epoch_time = time.perf_counter() - epoch_time_start
        # Print training progress
        if epoch % 1 == 0:
            print(f'Epoch [{epoch}/{num_epochs}]: Expected Reward Gradient: {epoch_expected_reward_gradient:.4f}, Mean Reward: {epoch_reward:.4f}, No. Samples: {tot_num_samples}, Baseline: {baseline:.4f}, Time: {epoch_time:.2f} seconds')
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "expected_reward_gradient": epoch_expected_reward_gradient,
            "mean_reward": epoch_reward,
            "time": epoch_time
        })
        # Save the checkpoint after the current epoch
        save_checkpoint(model, optimizer, epoch, epoch_reward, 0, epoch_expected_reward_gradient, save_checkpoint_path)
    time_end = time.perf_counter()
    print(f'Total training time: {time_end - time_start:.2f}s')
if __name__ == "__main__":
    main()

