from src.rotated_surface_code import RotatedCode
from src.graph_representation import get_syndrome_graph
from src.mwpm_prediction import compute_mwpm_reward
from src.gnn_model import EdgeWeightGNN, sample_weights_get_log_probs
from src.utils import test_model, save_checkpoint
import torch
import numpy as np
# python test_edge_predictor.py

def main():
    rates = np.arange(0.01, 0.21, 0.01)     # physical error rates
    d = 3
    code = RotatedCode(d)
    print(f'Testing d = {d}.')
    test_set_size = int(1e5)

    hidden_channels_GCN = [32, 64, 128, 256]
    hidden_channels_MLP = [512, 256, 128, 64, 32]

    model = EdgeWeightGNN(hidden_channels_GCN = hidden_channels_GCN, hidden_channels_MLP = hidden_channels_MLP)
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    # Check for checkpoint and load if available
    # generate a unique name to not overwrite other models
    name = 'd_3_250407_170810_resume'
    accuracy_file = 'accuracy_code_capacity/' + name + '_accuracy.csv'
    checkpoint_path = 'saved_models_code_capacity/' + name + '.pt'
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
        # initiate the dataset:
        test_set = []
        test_n_trivials = 0
        for _ in range(test_set_size):
            graph = get_syndrome_graph(code, p)
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

