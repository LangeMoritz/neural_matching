'''
Class for simulating the rotated surface code, with methods for 
generating random errors from depolarizing noise, converting errors to 
syndromes, and decoding them with MWPM using the PyMatching library.
'''
import numpy as np

class RotatedCode:
    def __init__(self, size: int):

        """
        Arguments:
            size: Lattice size L of the code; number of data qubits per row
        """

        # Define object parameters
        self.size = size
        self.n_qubits = int(size**2)

        # Create a shared check matrix for X and Z stabilizers
        # Since check matrix for X and Z stabilizers are identical, only 
        # create one and use for both to use less memory for large codes
        self.H = rot_surface_code_check_matrix(size)

# Helper function to produce check matrix for X stabilizers
def rot_surface_code_check_matrix(L):
    """
    Check matrix for the X stabilizers of a rotated 
    surface code with lattice size L.
    Assumes northwesternmost plaquette is a 
    weight-4 X stabilizer.
    Can also be seen as Z stabilizers with
    Z stabilizer in northwesternmost plaquette.
    """
    n = L**2
    m = L**2 - 1
    H = np.zeros([int(m/2),n], dtype=np.uint8)
    stabilizer_count = 0

    # Store boundary and inner plaquettes separately for 
    # more intuitive indexing of the stabilizers: 
    # row major order, starting with weight-2
    # stabilizers on north boundary and ending 
    # with weight-2 stabilizers on south boundary 
    north_stabilizers = np.zeros([int((L-1)/2),n], dtype=np.uint8)
    inner_stabilizers = np.zeros([int(m/2)-(L-1),n], dtype=np.uint8)
    south_stabilizers = np.zeros([int((L-1)/2),n], dtype=np.uint8)
    north_count = inner_count = south_count = 0
    H_list = [north_stabilizers, inner_stabilizers, south_stabilizers]

    # Iterate over plaquette rows 
    for ip in range( L-1 ):
        
        # Iterate over plaquette columns
        for jp in range( L-1 ):
            
            # Get contiguous index of plaquette
            indp = get_contig_index(L-1,ip,jp)

            # Get indices of adjacent data qubits
            nw = indp + ip
            ne = nw + 1
            sw = indp + L + ip
            se = sw + 1

            # Weight 4-stabilizers
            # Even rows, X-stabilizers
            if (ip % 2 == 0) and (jp % 2 == 0):
                inds = [nw, ne, sw, se]
                # Store in list of weight-4 stabilizers
                stabilizer_type = 1
                stabilizer_count = inner_count
                inner_count += 1

            # Weight 4-stabilizers
            # Odd rows, X-stabilizers
            elif (ip % 2 == 1) and (jp % 2 == 1): 
                inds = [nw, ne, sw, se]
                # Store in list of weight-4 stabilizers
                stabilizer_type = 1
                stabilizer_count = inner_count
                inner_count += 1

            # Weight 2-stabilizers
            # North boundary, X-stabilizers
            elif (ip == 0) and (jp % 2 == 1):
                # Corresponds to nw and ne of the Z-plaquette underneath
                inds = [nw, ne]
                # Store in list of weight-2 stabilizers on 
                # north boundary
                stabilizer_type = 0
                stabilizer_count = north_count
                north_count += 1

            # Weight 2-stabilizers
            # South boundary, X-stabilizers
            elif (ip == L - 2) and (jp % 2 == 0):
                # Corresponds to sw and se of the Z-plaquette above
                inds = [sw, se]
                # Store in list of weight-2 stabilizers on 
                # south boundary
                stabilizer_type = 2
                stabilizer_count = south_count
                south_count += 1
            
            # Ignore Z-plaquettes
            else:
                continue

            # Update check matrix entry in list of stabilizers
            H_list[stabilizer_type][stabilizer_count,inds] = 1

            # Re-organize stabilizers in H to row major order,
            # starting and ending with north and south boundaries
            H = np.concatenate(H_list, axis=0)

    return H

# Helper function to re-order entries in error strings,
# corresponding to rotating the code 90 degrees counter-clockwise
def  x_to_z_qubit_indices(error_string):
    """
    Re-orders the entries of an error string from the
    representation where the northwesternmost plaquette is a 
    weight-4 X-stabilizer to the representation where it is a 
    Z-stabilizer.
    (i.e. where the code is rotated 90 degrees counter-clockwise)
    """
    L = int(np.sqrt(len(error_string)))
    error_matrix_z = np.array(error_string).reshape(L,L)
    error_matrix_z = np.rot90(error_matrix_z)
    return error_matrix_z.flatten()

# Helper function to convert 2D array indices to contiguous index
def get_contig_index(L,ix,jx):
    """
    Get index in row major ordering of 
    lattice from row and column indices,
    for a lattice of size L.
    """
    if (ix >= L) or (jx >= L):
        print("Index out of bounds")
        return None
    ind = jx + L*ix
    return ind