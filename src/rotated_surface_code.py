'''
Class for simulating the rotated surface code, with methods for 
generating random errors from depolarizing noise, converting errors to 
syndromes, and decoding them with MWPM using the PyMatching library.
'''
import numpy as np
from typing import Optional, Union
import pymatching

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

    def get_syndrome_string(self, error: np.ndarray):
        # Flatten error if given as a 2D matrix
        error_string = error.flatten()

        # Create separate binary strings for X and Z errors
        error_string_x = (error_string == 1) + (error_string == 2)
        error_string_z = (error_string == 3) + (error_string == 2)
        # Re-order X-errors to get correct qubit 
        # indices for Z-stabilizers
        error_string_x = x_to_z_qubit_indices(error_string_x)

        # Since they are identical, use same check matrix for both Z and X

        H_check = self.H
        
        # Create separate syndrome strings for X and Z stabilizers
        # Note: X errors are checked by Z stabilizers and vice versa
        syndrome_string_x = H_check@error_string_z % 2
        syndrome_string_z = (H_check@error_string_x % 2) * 3

        return syndrome_string_x, syndrome_string_z

    def syndrome_string_to_matrix(self, syndrome_string_x, syndrome_string_z):
        
        # Create separate syndrome matrices for X and Z stabilizers (sped up with numpy slicing
        # instead of calling syndrome_string_to_matrix)
        M = self.size + 1
        syndrome_matrix_X = np.zeros((M, M), dtype=np.uint8)
        syndrome_string_X = syndrome_string_x.reshape(M, - 1)
        syndrome_matrix_X[::2, 2::2] = syndrome_string_X[::2]
        syndrome_matrix_X[1::2, 1:M - 2:2] = syndrome_string_X[1::2]

        syndrome_matrix_Z = np.zeros((M, M), dtype=np.uint8)
        syndrome_string_Z = syndrome_string_z.reshape(M, - 1)
        syndrome_matrix_Z[::2, 2::2] = syndrome_string_Z[::2]
        syndrome_matrix_Z[1::2, 1:M - 2:2] = syndrome_string_Z[1::2]

        # Combine syndrome matrices where 1 entries 
        # correspond to x and 3 entries to z defects
        syndrome_matrix = syndrome_matrix_X + np.rot90(syndrome_matrix_Z,-1)
        
        # Return the syndrome matrix
        return syndrome_matrix

    def get_syndrome(self, error: np.ndarray) -> np.ndarray:
        """
        Determines the syndrome consistent with an input error, given
        as a numpy array. error can either be a matrix corresponding to 
        errors on data qubits on the lattice, or a string of errors on 
        qubits in row major order. Entries in error should be (0,1,2,3)
        for (I,X,Y,Z) errors, respectively. The syndrome is a (L+1, L+1)
        matrix with 1 for x and 3 for Z type stabilisers, resp.
        """
        syndrome_string_x, syndrome_string_z = self.get_syndrome_string(error) 
        syndrome_matrix = self.syndrome_string_to_matrix(syndrome_string_x, syndrome_string_z)
        # Store syndrome matrix as object parameter
        self.syndrome = syndrome_matrix
        # Return the syndrome matrix
        return syndrome_matrix

    def get_eq_class(self, error: np.ndarray) -> int:
        """
        Determines the equivalence class of an 
        error chain on the rotated surface code, defined
        as a matrix of errors on the data qubits or
        as the corresponding flattened, by counting
        parity of Z-errors on the western boundary
        and X-errors on the north boundary.

        Inspired by/taken from _define_equivalence_class 
        function from EWD_QEC github repo.
        """
        # Reshape to a 2D matrix if error is given 
        # as flattened array
        L = self.size
        if error.ndim == 1:
            error_matrix = error.reshape(L,L)
        else:
            error_matrix = error


        # Count number of Z-errors on west edge
        Z_count = np.count_nonzero( 
            (error_matrix[:,0] == 3) + (error_matrix[:,0] == 2) ) 
        # Count number of X-errors on north edge
        X_count = np.count_nonzero(
            (error_matrix[0,:] == 1) + (error_matrix[0,:] == 2) ) 
        # Determine equivalence class from parity of Z-errors on west edge
        # and X-errors on north edge
        return np.array([X_count % 2, Z_count % 2]) # Equivalence class X ([1, 0])

    def init_matching(self):
        """
        Creates separate pymatching Matching objects for the
        Z and X stabilizers of the rotated surface code.
        """
        # Since Hx and Hz are identical, and we instead rotate
        # the syndrome to separate X from Z, only use one matching and one H.
        self.matching = rot_surface_code_matching(L = self.size, H = self.H)
        return

    def get_MWPM_correction(self, error: np.ndarray) -> np.ndarray:
        """
        Determines the MWPM correction of an error chain by constructing
        the syndrome consistent with the error and decoding the 
        syndrome using pymatching.
        """
        # Flatten error if given as a 2D matrix
        error_string = error.flatten() 

        # Create separate binary strings for X and Z errors
        error_string_x = (error_string == 1) + (error_string == 2)
        error_string_z = (error_string == 3) + (error_string == 2)
        # Re-order X-errors to get correct qubit 
        # indices for Z-stabilizers
        error_string_x = x_to_z_qubit_indices(error_string_x)
        H_check = self.H
        # Create separate syndrome strings for X and Z stabilizers
        # Note: X errors are checked by Z stabilizers and vice versa
        syndrome_string_x = H_check@error_string_z % 2
        syndrome_string_z = H_check@error_string_x % 2
        # Determine the MWPM corrections from the pymatching decoder
        # Note: Z correction is decoded from X syndrome and vice versa
        correction_z = self.matching.decode(syndrome_string_x)
        correction_x = self.matching.decode(syndrome_string_z)
        # Convert the correction strings to matrices
        correction_matrix_z = error_string_to_matrix(correction_z)
        correction_matrix_x = error_string_to_matrix(correction_x)
        # Correct indices for re-ordering of X errors to match Z stabilizers
        correction_matrix_x = np.rot90(correction_matrix_x,-1) 
        # Create combined correction matrix, with entries (0,1,2,3)
        # corresponding to errors (I,X,Y,Z)
        correction_matrix = correction_matrix_x + correction_matrix_z*3
        correction_matrix[correction_matrix == 4] = 2
        # Return combined correction matrix
        return correction_matrix

    def generate_error_chain(self, p: float,
            random_seed: Optional[int] = None) -> np.ndarray:
        """
        Produces a chain of errors on the physical qubits:
        0 : no error, 1 : X error, 2: Y error, 3: Z error
        """
        n_qubits = self.n_qubits
        # eta = 10
        # p_y = p / (1 + 1 / eta)
        # p_x = p_z = p_y / (2 * eta) # Y biased noise
        p_x = p_y = p_z = p/3 # Unbiased noise
        noise = np.zeros(n_qubits, dtype = np.uint8)
        draws = np.random.random(n_qubits)
        noise[(draws <= p_x)] = 1
        noise[(p_x < draws) & (draws <= p_x+p_y)] = 2
        noise[(p_x+p_y < draws) & (draws <= p_x+p_y+p_z)] = 3

        return noise

    def measurement_errors(self, string_x, string_z, p):
        """
        Function for introducing measurement errors on the stabilisers. 
        With a probability p, flip the ancilla qbits. Takes a syndrome
        in X- and Z-string representation. 
        """
        n_x_stabilisers = string_x.shape[0] # = (code_size ** 2 - 1) / 2
        n_z_stabilisers = string_z.shape[0] # number of stabilisers of each kind
        draws_x = np.random.random(n_x_stabilisers)
        draws_z = np.random.random(n_z_stabilisers)
        errors_x = np.zeros(n_x_stabilisers)
        errors_z = np.zeros(n_z_stabilisers)
        errors_x[(draws_x <= p)] = 1 # introduce random errors
        errors_z[(draws_z <= p)] = 3
        string_x = (string_x + errors_x) % 2 # compare those errors with syndromes and flip
        string_z = ((string_z + errors_z) % 2) * 3
        return string_x, string_z
    
    def alternate_error_chain(self, error_chain, p):
        """
        From the original error chain, introduce even more errors.
        """
        n_qubits = self.n_qubits
        p_x = p_y = p_z = p/3 # Unbiased noise

        # new error chain: 
        noise = np.zeros(n_qubits, dtype = np.uint8)
        draws = np.random.random(n_qubits)
        noise[(draws <= p_x)] = 1
        noise[(p_x < draws) & (draws <= p_x+p_y)] = 2
        noise[(p_x+p_y < draws) & (draws <= p_x+p_y+p_z)] = 3
        # compare with old one: 
        error_chain_alternated = np.bitwise_xor(noise, error_chain)
        return error_chain_alternated.astype(np.uint8)
    
    def plot_syndrome(self, syndrome: np.ndarray,
            error: Optional[np.ndarray] = None, 
            show_syndrome: Optional[bool] = True,
            fill_X: Optional[bool] = True,
            fill_Z: Optional[bool] = True,
            X_defect_color: Optional[str] = None,
            Z_defect_color: Optional[str] = None,
            file_name: Optional[str] = None,
            figure_size: Optional[int] = None,
            title: Optional[str] = None,
            fig = None, 
            ax = None):
        L = self.size
            

        # Define plot parameters
        lw = 1 # Linewidth of lines between qubits
        qubit_size = 10 # Markersize of qubits
        defect_size = 10 # Markersize of defects
        
        X_plaquette_color = '#fc8d62' # Same color as x stabilizers and errors, but more transparent
        Z_plaquette_color = '#8da0cb' # Same color as z stabilizers and errors, but more transparent
        if X_defect_color is None:
            X_defect_color = '#8da0cb' #Black
        if Z_defect_color is None:
            Z_defect_color = '#fc8d62' # Red
        # https://coolors.co/005aa4

        # Inner plaquette edges
        for x, y in zip(range(L), range(L)):
            ax.plot([x,x],[0,L-1], color='black', linewidth=lw)
            ax.plot([0,L-1], [y,y], color='black', linewidth=lw)

        # Boundary plaquette edges
        # Generate points for one semicircle centered at (0,0)
        x_center = 0.
        y_center = 0.
        x, y = _get_semicircle(x_center, y_center)
        # Translate coordinates to each border
        for ix in range(int(L/2)):
            # North boundary
            ax.plot(x+1.5+ix*2, y+L-1, color='black', linewidth=lw)
            # South boundary
            ax.plot(x+0.5+ix*2, -y, color='black', linewidth=lw)
            # West boundary
            ax.plot(-y, x+1.5+ix*2, color='black', linewidth=lw)
            # East boundary
            ax.plot(y+L-1, x+0.5+ix*2, color='black', linewidth=lw)

        if fill_X:
            _fill_X(ax, L, X_plaquette_color)

        if fill_Z:
            _fill_Z(ax, L, Z_plaquette_color)

        # Data qubits
        dataX, dataY = np.meshgrid(range(L),range(L))
        ax.plot(dataX,dataY,'o', color='black', 
            markerfacecolor='white', markersize=qubit_size)

        # Syndrome
        # Reshape to a 2D matrix
        if syndrome.ndim == 1:
            syndrome = syndrome.reshape(L+1,L+1)
        # Get defect types
        defect_types = self._get_defect_types(syndrome)
        # Get lattice coordinates of defects 
        # Matrix indices to coords (flipud), x is col and y is row (fliplr)
        # (- 0.5) to shift from data qubit lattice to syndrome lattice
        defect_types = np.flipud(defect_types)[np.flipud(defect_types) != 0]
        defect_coords = np.fliplr(np.argwhere(np.flipud(syndrome))) - 0.5

        if show_syndrome:
            for defect, (x,y) in zip(defect_types,defect_coords):
                if defect == 1:
                    # X defect
                    ax.plot(x, y, 'o', color=X_defect_color, markersize=defect_size,
                        markeredgecolor=None)
                elif defect == 3:
                    # Z defect
                    ax.plot(x, y, 'o', color=Z_defect_color, markersize=defect_size,
                        markeredgecolor=None)

        # Error
        if error is not None:
            _plot_error(ax, L, error, defect_size)

        # Title:
        if title is not None: 
            ax.title(title)

        # Set axis properties
        ax.axis('off')
        ax.axis('square')

        # Save and clear figure
        if file_name is not None:
            fig.savefig(file_name, bbox_inches = 'tight')
        return

    def _get_defect_types(self, syndrome):
        L = self.size
        defect_indices = np.where(syndrome.flatten() != 0)[0]
        defect_types = np.zeros((L+1)**2)

        # Iterate over defects
        # Determine if X stabilizer, otherwise must be Z stabilizer
        for defect_index in defect_indices:
            defect_is_x_stabilizer = 0
            ix = int(defect_index/(L+1))
            jx = defect_index % (L+1)

            # Ignore west and east boundary, as they must be Z stabilizers
            if (0 < jx < L):
                # Even rows
                if ix % 2 == 0:
                    # Even columns
                    if jx % 2 == 0:
                        defect_is_x_stabilizer = 1 
                # Odd rows
                else:
                    # Odd columns
                    if jx % 2 == 1:
                        defect_is_x_stabilizer = 1 

            # === Get X and Z features ===
            if defect_is_x_stabilizer == 1:
                defect_types[defect_index] = 1 # X stabilizer
            else:
                defect_types[defect_index] = 3 # Z stabilizer
        return defect_types.reshape((L+1,L+1))

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

# Helper function to convert syndrome strings to matrix representation
def syndrome_string_to_matrix(s):
    """
    Creates a (L+1,L+1) matrix of syndromes
    on the rotated surface code of size L
    from a syndrome bit string s in row 
    major order. 
    Assumes syndrome corresponds to X stabilizer
    measurements, where the northwesternmost 
    plaquette is a weight-4 X stabilizer.
    Can also be seen as Z stabilizers with
    Z stabilizer in northwesternmost plaquette.
    """
    L = int(np.sqrt(len(s)*2 + 1)) # Calculate lattice size

    # If syndrome is generated from PyMatching's add_noise()
    # Remove boundary nodes (last two elements)
    if len(s) != int((L**2-1)/2):
        s = s[:-2] 


    plaquette_matrix = np.zeros([L+1, L+1], dtype=np.uint8)
    # Inner plaquettes: plaquette_matrix[1:-1,1:-1]
    # North boundary half plaquettes: plaquette_matrix[0,1:-1]
    # South boundary half plaquettes: plaquette_matrix[-1,1:-1]

    # Iterate over plaquette rows 
    for ip in range( L-1 ):
        
        # Iterate over plaquette columns
        for jp in range( L-1 ):
            
            # Get contiguous index of plaquette
            indp = get_contig_index(L-1,ip,jp)
            
            # Even rows, X-stabilizers
            if (ip % 2 == 0) and (jp % 2 == 0)\
            and s[int((L-1)/2):-int((L-1)/2)][int(indp/2)] != 0:
                plaquette_matrix[1:-1,1:-1][ip][jp] = 1
                
            # Weight 4-stabilizers
            # Odd rows, X-stabilizers
            elif (ip % 2 == 1) and (jp % 2 == 1)\
            and s[int((L-1)/2):-int((L-1)/2)][int(indp/2)] != 0:
                plaquette_matrix[1:-1,1:-1][ip][jp] = 1

            # Weight 2-stabilizers
            # North boundary, X-stabilizers
            elif (ip == 0) and (jp % 2 == 1)\
            and s[:int((L-1)/2)][int(indp/2)] != 0:
                plaquette_matrix[0,1:-1][jp] = 1
            # indp/2 rounded down gives corresponding index in s 
            # for weight-2 plaquettes on north boundary
            # (after removing other plaquettes from s)

            # Weight 2-stabilizers
            # South boundary, X-stabilizers
            elif (ip == L - 2) and (jp % 2 == 0)\
            and s[int((L-1)/2):][int(indp/2)+int((L-1)/2)] != 0:
                plaquette_matrix[-1,1:-1][jp] = 1
            
            # Ignore Z-plaquettes
            else:
                continue

    return plaquette_matrix

# Helper function for creating pymatching Matching object from a 
# parity check matrix
def rot_surface_code_matching(L, H, p=None):
    """
    Creates Matching object for a rotated 
    surface code with lattice size L with
    error probability p, for one set of 
    stabilizers (X or Z).
    By default, p is set to None. To use the 
    pymatching method add_noise(), p needs to 
    be specified.
    Assumes northwesternmost plaquette is a 
    weight-4 X stabilizer.
    Can also be seen as Z stabilizers with
    Z stabilizer in northwesternmost plaquette.
    H specifies which check matrix to use (i.e. 
    whether the matching is for X or Z stabilizers)
    """

    # Create check matrix using helper function
    # Update: Use Hx and Hz, no need to create new
    # H = rot_surface_code_check_matrix(L) 

    # Start from empty Matching object
    try:
        m = pymatching.Matching() 
    except:
        print('PyMatching module not imported')
    # Get number of X-stabilizers (nodes) and data qubits (edges)
    num_nodes, num_edges = H.shape 

    # Indices for boundary nodes
    boundary_node1 = num_nodes
    boundary_node2 = num_nodes+1
    for qubit_index in range(num_edges):
        
        # Select nodes for qubits with two stabilizers
        if np.sum(H[:,qubit_index]) == 2:
            # Get indices of connected stabilizers
            node1, node2 = np.where(H[:,qubit_index] == 1)[0]

        # Select nodes for qubits with one stabilizer, 
        # connect to boundary nodes
        elif np.sum(H[:,qubit_index]) == 1:
            # Get index of connected stabilizer
            node1 = np.where(H[:,qubit_index] == 1)[0]

            # If stabilizer has weight 2
            if np.sum(H[node1,:]) == 2:
                # Connect qubit to boundary node 1
                node2 = boundary_node1

            # If stabilizer has weight 4
            elif np.sum(H[node1,:]) == 4:
                # Get list of first nodes in already existing edges
                existing_edges_node1 = [item[0] for item in m.edges()]
                # If connected stabilizer is already connected 
                # to boundary node 1
                if node1 in existing_edges_node1:
                    # Connect to boundary node 2
                    node2 = boundary_node2
                # Else, connect to boundary node 1
                else:
                    node2 = boundary_node1

            else:
                print(
                    ("Error: rows in check matrix must have "
                    "exactly two or four non-zero elements"))
                return

        else:
            print(
                ("Error: columns in check matrix must have "
                "exactly one or two non-zero elements"))
            return

        # Add the edge 
        m.add_edge(node1, node2, 
                fault_ids = qubit_index, error_probability = p)

    # Specify the boundary nodes
    m.set_boundary_nodes({boundary_node1, boundary_node2})
    return m

def error_string_to_matrix(error):
    """
    Creates a (L,L) matrix of errors/corrections
    on the rotated surface code of size L
    from an error/correction bit string e 
    in row major order.
    """
    L = int(np.sqrt(len(error)))
    error_matrix = np.zeros([L, L], dtype=np.uint8)
    error_matrix = np.reshape(error, error_matrix.shape)
    return error_matrix

def num_to_class(num: Union[int, np.ndarray]) -> Union[str, np.ndarray]:
    # Helper function for converting an integer, representing
    # an equivalence class, error type, defect type, or similar,
    # to the corresponding class.
    #
    # Convention used is 
    # 0 <-> I
    # 1 <-> X
    # 2 <-> Y
    # 3 <-> Z
    if type(num) is np.ndarray:
        str_arr = np.array(num.shape, dtype=str)
        for ix, n in enumerate(num):
            if n > 3 or n < 0:
                raise ValueError("Must be an integer between 0 and 3")
            if n == 0:
                str_arr[ix] = 'I'
            elif n == 1:
                str_arr[ix] = 'X'
            elif n == 2:
                str_arr[ix] = 'Y'
            elif n == 3:
                str_arr[ix] = 'Z'
        return str_arr


    if num > 3 or num < 0:
        raise ValueError("Must be an integer between 0 and 3")
    if num == 0:
        return 'I'
    elif num == 1:
        return 'X'
    elif num == 2:
        return 'Y'
    elif num == 3:
        return 'Z'
     

def _get_semicircle(x_center, y_center, 
        a_start=0., a_end=np.pi, radius=0.5, 
        semicircle_num=50):    
    '''Helper function used when plotting the weight-2 stabilizers'''
    a = np.linspace(a_start, a_end, semicircle_num)
    x = np.cos(a)*radius
    y = np.sin(a)*radius
    return x + x_center, y + y_center

def _fill_X(ax, L, X_color):
    '''Helper function that fills X-plaquettes in surface code plot'''

    # Fill inner X plaquettes
    # Iterate over even rows and columns
    alpha = 1.0
    for iy in range(int(L/2)): 
        for ix in range(int(L/2)):
            # Get coordinates of northwest corner of plaquette
            x_nw = ix*2
            y_nw = L-1 - iy*2
            # Calculate coordinates of plaquette corners (clockwise)
            x_corners = np.array([x_nw, x_nw+1, x_nw+1, x_nw])
            y_corners = np.array([y_nw, y_nw, y_nw-1, y_nw-1])
            ax.fill(x_corners, y_corners, color=X_color, alpha = alpha)
            # Fill the plaquette to the southeast in the next row
            ax.fill(x_corners + 1, y_corners - 1, color=X_color, alpha = alpha)

    x_center = 0.
    y_center = 0.
    # Fill boundary X plaquettes 
    x, y = _get_semicircle(x_center, y_center)
    for ix in range(int(L/2)):
        # North boundary
        ax.fill(x+1.5+ix*2, y+L-1, color=X_color, alpha = alpha)
        # South boundary
        ax.fill(x+0.5+ix*2, -y, color=X_color, alpha = alpha)


def _fill_Z(ax, L, Z_color):
    '''Helper function that fills Z-plaquettes in surface code plot'''
    alpha = 1.0
    # Fill inner Z plaquettes
    # Iterate over even rows and columns
    for iy in range(int(L/2)): 
        for ix in range(int(L/2)):
            # Get coordinates of northwest corner of plaquette
            x_nw = 1 + ix*2
            y_nw = L-1 - iy*2
            # Calculate coordinates of plaquette corners (clockwise)
            x_corners = np.array([x_nw, x_nw+1, x_nw+1, x_nw])
            y_corners = np.array([y_nw, y_nw, y_nw-1, y_nw-1])
            ax.fill(x_corners, y_corners, color=Z_color, alpha = alpha)
            # Fill the plaquette to the southwest in the next row
            ax.fill(x_corners - 1, y_corners - 1, color=Z_color, alpha = alpha)

    x_center = 0.
    y_center = 0.
    # Fill boundary Z plaquettes 
    x, y = _get_semicircle(x_center, y_center)
    for ix in range(int(L/2)):
        # West boundary
        ax.fill(-y, x+1.5+ix*2, color=Z_color, alpha = alpha)
        # East boundary
        ax.fill(y+L-1, x+0.5+ix*2, color=Z_color, alpha = alpha)

def _plot_error(ax, L, error, defect_size):
    '''Helper function for plotting errors in the rotated surface code'''
    # defect_size: size of markers
    # Reshape to a 2D matrix
    if error.ndim == 1:
        error = error.reshape(L,L)
    # Get lattice coordinates of errors 
    # Matrix indices to coords (flipud), x is col and y is row (fliplr)
    flipped_error = np.flipud(error)
    error_coords = np.fliplr(np.argwhere(flipped_error))
    # Get error types, flip to get same order as error_coords
    error_types = flipped_error[flipped_error != 0]
    for (x,y),e in zip(error_coords, error_types):
        if e == 1:
            ax.text(x, y, 'X', fontsize = 'large') 
        if e == 2:
            ax.text(x, y, 'Y', fontsize = 'large') 
        if e == 3:
            ax.text(x, y, 'Z', fontsize = 'large') 
        
        # plt.plot(x, y, 'o', color=col, markersize=defect_size)
        # Plot class index of error