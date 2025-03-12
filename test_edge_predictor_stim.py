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

    model = EdgeWeightGNN_stim()
    model.eval()
    # Check for checkpoint and load if available
    # generate a unique name to not overwrite other models
    name = ("d_" + str(d))
    accuracy_file = 'accuracies/stim_gcn_32_64_128_mlp_128_64_32_memory_x/' + name + '_accuracy.csv'
    name = "d_" + str(d) + "_d_t_" + str(d_t) + "_p_0p001" 
    checkpoint_path = 'saved_models/stim_gcn_32_64_128_mlp_128_64_32_memory_x/' + name + '.pt'
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

