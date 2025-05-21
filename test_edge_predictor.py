from src.rotated_surface_code import RotatedCode
from src.graph_representation import get_syndrome_graph
from src.mwpm_prediction import compute_mwpm_rewards_batched
from src.gnn_model import EdgeWeightGNN_batch, sample_weights_get_log_probs
from src.utils import test_model, save_checkpoint
from torch_geometric.data import Batch
import torch
import numpy as np
# python test_edge_predictor.py

def main():
    rates = np.arange(0.01, 0.21, 0.01)     # physical error rates
    d = 5
    code = RotatedCode(d)
    print(f'Testing d = {d}.')
    test_set_size = int(1e5)
    batch_size = int(1e3)
    n_test_batches = int(test_set_size / batch_size)

    device = torch.device("cpu")

    hidden_channels_GCN = [32, 64, 128, 256]
    hidden_channels_MLP = [512, 128, 64]

    model = EdgeWeightGNN_batch(hidden_channels_GCN = hidden_channels_GCN, hidden_channels_MLP =hidden_channels_MLP)
    model.to(device)
    model.eval()
    # Check for checkpoint and load if available
    # generate a unique name to not overwrite other models
    name = 'd5_Y_biased_4462389_resume_4466838'
    accuracy_file = 'accuracies/alvis_code_capacity/' + name + '_1e5_eta10.csv'
    checkpoint_path = 'saved_models/alvis_code_capacity/' + name + '.pt'
    start_epoch = 0
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']  # Get the epoch from checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        print(f"Checkpoint loaded, continuing from epoch {start_epoch} with {checkpoint_path}.")
    except FileNotFoundError:
        print("No checkpoint found, testing from scratch.")

    logical_accs = []
    for p in rates:
        test_acc = 0 
        for _ in range(n_test_batches):
            # initiate the dataset:
            test_set = []
            test_n_trivials = 0
            for _ in range(batch_size):
                graph = get_syndrome_graph(code, p)
                if not graph == None:
                   test_set.append(graph)
                else: 
                    test_n_trivials += 1
            graph_batch = Batch.from_data_list(test_set).to(device)

            n_nontrivial_test_samples = len(test_set)
            n_trivial_test_samples = batch_size - n_nontrivial_test_samples

            with torch.no_grad():
                # Forward pass: Get edge weights and number of real and boundary nodes
                edge_index, edge_weights_mean, batch, graph_info = model(graph_batch.x,
                                                                  graph_batch.edge_index,
                                                                  graph_batch.edge_attr,
                                                                  graph_batch.batch)
            rewards_nontrivial = compute_mwpm_rewards_batched(edge_index,
                                         edge_weights_mean.reshape(1, -1),
                                         graph_info,
                                         graph_batch.y)
            test_acc_nontrivial = (rewards_nontrivial.mean() + 1) / 2
            num_corr_nontrivial = test_acc_nontrivial * n_nontrivial_test_samples
            test_acc += (num_corr_nontrivial + n_trivial_test_samples) / test_set_size
        print(f'Physical error rate: {p:.4f}, Logical failure rate: {1 -test_acc:.4f}, Logical accuracy: {test_acc:.4f}')
        logical_accs.append(test_acc)
    np.savetxt(accuracy_file, np.array(logical_accs), delimiter=",")
if __name__ == "__main__":
    main()

