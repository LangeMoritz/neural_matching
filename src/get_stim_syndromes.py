'''Package with functions for creating stim syndromes.'''
import numpy as np
import stim

def initialize_simulations(code_size, d_t, error_rate):
    circuit = stim.Circuit.generated(
                            "surface_code:rotated_memory_x",
                            rounds = d_t,
                            distance = code_size,
                            after_clifford_depolarization = error_rate,
                            after_reset_flip_probability = error_rate,
                            before_measure_flip_probability = error_rate,
                            before_round_data_depolarization = error_rate)
    compiled_sampler = circuit.compile_detector_sampler()

    # get detector coordinates:
    detector_coordinates = circuit.get_detector_coordinates()
    # get coordinates of detectors (divide by 2 because stim labels 2d grid points)
    # coordinates are of type (d_west, d_north, hence the reversed order)
    detector_coordinates = np.array(list(detector_coordinates.values()))
    # rescale space like coordinates:
    detector_coordinates[:, : 2] = detector_coordinates[:, : 2] / 2
    # convert to integers
    detector_coordinates = detector_coordinates.astype(np.uint8)
    # syndrome mask
    sz = code_size + 1
    syndrome_x = np.zeros((sz, sz), dtype=np.uint8)
    syndrome_x[::2, 1 : sz - 1 : 2] = 1
    syndrome_x[1::2, 2::2] = 1
    syndrome_z = np.rot90(syndrome_x) * 3
    syndrome_mask = np.dstack([syndrome_x + syndrome_z] * (d_t + 1))
    return compiled_sampler, syndrome_mask, detector_coordinates

def stim_to_syndrome_3D(syndrome_mask, detector_coordinates, detection_events_list):
    '''
    Converts a stim detection event array to a syndrome grid. 
    1 indicates a violated X-stabilizer, 3 a violated Z stabilizer.
    '''
    mask = np.repeat(syndrome_mask[None, ...], detection_events_list.shape[0], 0)
    syndrome_3D = np.zeros_like(mask)
    syndrome_3D[:, detector_coordinates[:, 1], detector_coordinates[:, 0],
                detector_coordinates[:, 2]] = detection_events_list
    # convert X (Z) stabilizers to 1(3) entries in the matrix
    syndrome_3D[np.nonzero(syndrome_3D)] = mask[np.nonzero(syndrome_3D)]
    # return as [n_shots, x_coordinate, z_coordinate, time]
    return syndrome_3D
