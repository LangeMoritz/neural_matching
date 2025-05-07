from src.rotated_surface_code import RotatedCode
from src.graph_representation import get_syndrome_graph
from src.mwpm_prediction import compute_mwpm_reward, compute_mwpm_rewards_multiple_draws, compute_mwpm_rewards_multi_syndrome
from src.gnn_model import EdgeWeightGNN, sample_weights_get_log_probs
from src.utils import test_model, save_checkpoint
import torch
import argparse
import numpy as np
import os
import time
from datetime import datetime
# python -m cProfile -o figures_and_outputs/multi_syndromes.prof train_edge_predictor.py
# snakeviz figures_and_outputs/multi_syndromes.prof
import wandb
os.environ["WANDB_SILENT"] = "True"

def main():
    time_start = time.perf_counter()
    # Initialize the argument parser
    # parser = argparse.ArgumentParser(description="Train the neural network with specified parameters")
    # parser.add_argument('--d', type=int, required=True, help='The value of d')

    # # Parse the arguments
    # args = parser.parse_args()

    # # Access the value of d
    # d = args.d

    # # Get the SLURM job ID from the environment
    # job_id = os.environ.get('SLURM_JOB_ID', 'unknown')  # Default to 'unknown' if not running in SLURM
    d = 5
    p = [0.01, 0.05, 0.1, 0.15]  # physical error rates
    code = RotatedCode(d)
    print(f'Training d = {d}.')
    num_samples_per_epoch = int(1e2)
    num_draws_per_sample = int(100)
    tot_num_samples = 0
    stddev = torch.tensor(0.05, dtype=torch.float32)
    lr = 1e-4
    num_epochs = 200

    hidden_channels_GCN = [32, 64, 128, 256]
    hidden_channels_MLP = [512, 128, 64]

    model = EdgeWeightGNN(hidden_channels_GCN = hidden_channels_GCN, hidden_channels_MLP = hidden_channels_MLP)
    device = torch.device('cpu')
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Check for checkpoint and load if available
    load_checkpoint_path = '/Users/xlmori/Desktop/neural_matching/saved_models/vera_code_cap/d3_250425_163528_Y_biased_7026769_resume_7029590.Xpt'  # path of existing checkpoint
    # current_datetime = datetime.now().strftime("%y%m%d_%H%M%S")
    # name = 'd' + str(d) + '_' + current_datetime + '_' + job_id
    # save_checkpoint_path = f'saved_models_code_capacity/{name}.pt'
    
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

    # wandb.init(project="neural_matching_code_capacity", name = name, config = {
    #     "learning_rate": lr,
    #     "epochs": num_epochs,
    #     "num_samples_per_epoch": num_samples_per_epoch,
    #     "num_draws_per_sample": num_draws_per_sample,
    #     "stddev": stddev.item(),
    #     "d": d,
    #     "p": p,
    #     "hidden_channels_GCN": hidden_channels_GCN, 
    #     "hidden_channels_MLP": hidden_channels_MLP})

    baseline = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_time_start = time.perf_counter()
        optimizer.zero_grad()

        all_num_real_nodes = []
        all_num_boundary_nodes = []
        all_logical_classes = []
        all_log_probs = []

        # initiate the dataset:
        graph_list = []
        edge_shape = 0 
        while len(graph_list) < num_samples_per_epoch:
            p_one_sample = np.random.choice(p)
            graph = get_syndrome_graph(code, p_one_sample)
            if not graph == None:
                # Forward pass: Get edge weights and number of real and boundary nodes
                edge_index, edge_weights_mean, num_real_nodes, num_boundary_nodes = \
                    model(graph.x, graph.edge_index, graph.edge_attr)

                sampled_edge_weights, log_probs_per_sample = sample_weights_get_log_probs(edge_weights_mean,
                                                                                          num_draws_per_sample,
                                                                                          stddev)
                                                                                # shape: (num_draws_per_sample, num_edges)
                graph_list.append((edge_index.cpu().numpy(), sampled_edge_weights.cpu().numpy()))
                all_num_real_nodes.append(num_real_nodes)
                all_num_boundary_nodes.append(num_boundary_nodes)
                all_logical_classes.append(int(graph.y))
                all_log_probs.append(log_probs_per_sample)
                # if edge_index.shape[1] == edge_shape:
                #     print("Edge shape is the same as before.")
                # edge_shape = edge_index.shape[1]
        rewards = compute_mwpm_rewards_multi_syndrome(
                        graph_list,
                        all_num_real_nodes,
                        all_num_boundary_nodes,
                        all_logical_classes,
                        num_draws_per_sample,
                        num_samples_per_epoch
                    )  # shape: (num_draws, num_samples_per_epoch)
        # print(rewards)
        all_log_probs = torch.stack(all_log_probs, dim=1) # Shape: (num_draws_per_sample, num_samples_per_epoch)
        # The loss per draw and per edge is the log-probability times the reward - baseline
        all_expected_reward_gradients = - all_log_probs * (rewards - baseline)

        epoch_reward = torch.mean(rewards).item()
        epoch_expected_reward_gradient = torch.mean(all_expected_reward_gradients)
    
        epoch_expected_reward_gradient.backward()

        optimizer.step()  # Perform a single optimization step after accumulating gradients
        baseline = (baseline + epoch_reward) / 2
        tot_num_samples += num_samples_per_epoch

        epoch_time = time.perf_counter() - epoch_time_start
        # Print training progress
        if epoch % 1 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Log-Loss: {epoch_expected_reward_gradient:.4f}, Mean Reward: {epoch_reward:.4f}, No. Samples: {tot_num_samples}, Baseline: {baseline}, Time: {epoch_time:.2f} seconds')
        # Log to wandb
        # wandb.log({
        #     "epoch": epoch,
        #     "log_loss": epoch_log_loss,
        #     "mean_reward": epoch_reward,
        #     "time": epoch_time,
        #     "accuracy": train_acc,
        # })
        # # Save the checkpoint after the current epoch
        # save_checkpoint(model, optimizer, epoch, epoch_reward, train_acc, epoch_log_loss, save_checkpoint_path)

    time_end = time.perf_counter()
    print(f'Total training time: {time_end - time_start:.2f}s')
if __name__ == "__main__":
    main()

