from qecsim.graphtools import SimpleGraph, mwpm, mwpm_multi_syndromes
import numpy as np
from src.mwpm_prediction import compute_mwpm_rewards_multi_syndrome
import torch 
edge_index = torch.tensor([[0, 0, 1, 1, 2],
                           [1, 2, 2, 3, 3],])
edge_index = edge_index
edge_weights_draws = torch.tensor([[0.1, 0.3, 0, 0, 0],
                                   [0.4, 0.2, 0, 0, 0],
                                   [0.1, -0.2, 0, 0, 0],
                                   [0.5, 0.1, 0, 0, 0]])
syndrome_graphs = [(edge_index, edge_weights_draws)]
draws_per_syndrome = edge_weights_draws.shape[0]
# num_real_nodes_list = [1, 1, 1]
# num_boundary_nodes_list = [3, 3, 3]
# logical_classes = [0, 0, 0]
# rewards = compute_mwpm_rewards_multi_syndrome(syndrome_graphs, num_real_nodes_list, num_boundary_nodes_list, logical_classes)
# print(rewards)
matches = mwpm_multi_syndromes(syndrome_graphs, draws_per_syndrome)
for match in matches:
    print(match)

def describe_matches(matches):
    """
    Print which node pairs are matched for each draw of each syndrome.
    Args:
        matches: list of (num_draws, num_nodes) arrays
    """
    for s, match_array in enumerate(matches):
        print(f"Syndrome {s}:")
        for d, row in enumerate(match_array):
            seen = set()
            print(f"  Draw {d}:")
            for i, j in enumerate(row):
                if i < j and (j, i) not in seen:
                    print(f"    Node {i} ↔ Node {j}")
                    seen.add((i, j))
describe_matches(matches)