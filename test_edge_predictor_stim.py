from src.get_stim_syndromes import initialize_simulations
from src.gnn_model import EdgeWeightGNN_stim
from src.utils import test_model
from src.graph_representation_stim import get_syndrome_graph
import torch
import numpy as np
# python test_edge_predictor.py

def main():
    rates = np.arange(0.001, 0.006, 0.001)     # physical error rates
    d = 3
    d_t = d
    print(f'Testing d = {d}.')
    test_set_size = int(1e5)

    hidden_channels_GCN = [32, 64, 128, 256]
    hidden_channels_MLP = [512, 256, 128, 64, 32]

    model = EdgeWeightGNN_stim(hidden_channels_GCN = hidden_channels_GCN, hidden_channels_MLP = hidden_channels_MLP)
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    # Check for checkpoint and load if available
    # generate a unique name to not overwrite other models
    name = 'd_3_d_t_3_250407_101132'
    accuracy_file = 'accuracy/' + name + '_accuracy.csv'
    checkpoint_path = 'saved_models/' + name + '.pt'
    start_epoch = 0
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        start_epoch = checkpoint['epoch']  # Get the epoch from checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
        print(f"Checkpoint loaded, continuing from epoch {start_epoch}.")
    except FileNotFoundError:
        print("No checkpoint found, testing from scratch.")

    logical_accs = []
    for p in rates:
        compiled_sampler, syndrome_mask, detector_coordinates = initialize_simulations(d, d_t, p)

        # initiate the dataset:
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

        test_acc_nontrivial = test_model(model, n_nontrivial_test_samples, test_set)
        num_corr_nontrivial = test_acc_nontrivial * n_nontrivial_test_samples
        test_acc = (num_corr_nontrivial + n_trivial_test_samples) / test_set_size
        print(f'Physical error rate: {p:.4f}, Logical failure rate: {1 -test_acc:.4f}, Logical accuracy: {test_acc:.4f}')
        logical_accs.append(test_acc)
    np.savetxt(accuracy_file, np.array(logical_accs), delimiter=",")
if __name__ == "__main__":
    main()

