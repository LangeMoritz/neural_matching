'''
Class for decoding syndromes with graph neural networks, with methods for
training the network continuously with graphs from random sampling of errors 
as training data..
'''
import torch
import numpy as np
import os
import time
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from multiprocessing import Pool, cpu_count
import gc

from src.rotated_surface_code import RotatedCode
from src.graph_representation import get_torch_graph

class GNN_Decoder:
    def __init__(self, params = None):
        # Set default parameters
        self.params = {
            'model': {
                'class': None,
                'num_node_features': 4,
                'num_classes': 1,
                'loss': None,
                'initial_learning_rate': 0.01,
                'manual_seed': 12345
            },
            'graph': {
                'm_nearest_nodes': None,
                'num_node_features': 4,
                'power': 2
            },
            'cuda': False,
            'silent': False,
            'save_path': './',
            'save_prefix': None,
            'resumed_training_file_name': None
        }
        p = self.params

        # Use default parameters, update ones specified in input dictionary
        def update_params(params_dict, input):
            for key in input:
                if key in params_dict:
                    # Handle nested dictionaries recursively
                    if isinstance(params_dict[key], dict) and \
                        isinstance(input[key], dict):
                            update_params(params_dict[key], input[key])
                    elif not isinstance(params_dict[key], dict) and \
                        not isinstance(input[key], dict):
                            params_dict[key] = input[key]
        if params is not None:
            update_params(p, params)
        # Instantiate GNN model
        try:
            self.model = p['model']['class'](
                num_node_features = p['model']['num_node_features'],
                num_classes = p['model']['num_classes'],
                manual_seed = p['model']['manual_seed']
            )
        except TypeError:
            print('!!!!!!Input model must be a valid GNN class!!!!!!')

        if p['cuda']:
            # Use GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Move model to GPU if available
            self.model = self.model.to(device)

        # Create lists to store results from consecutive training loops
        self.clear_results()

        # Set code attribute to None
        self.code = None

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
            lr = p['model']['initial_learning_rate'])

    # Make class callable, running the model's forward method
    def __call__(self, **kwargs):
        self.model.eval() 
        return self.model(**kwargs)

    def clear_results(self):
        self.train_accuracies = []
        self.valid_accuracies = []
        self.train_losses = []
        self.valid_losses = []
        self.continuous_training_history = {
            'accuracy': [], # Accuracy for each training iteration
            'loss': [], # Average sample loss per training iteration
            'val_acc': [], # Accuracy of the validation set tested without trivial syndromes
            'num_samples_trained': 0 # Total dataset size since initialization
        }
    
    def save_attributes_to_file(self, prefix: str = None, suffix = ''):
        '''
        Save the current model, optimizer and training history to file.
        Path and file name are specified by instance attributes.
        By default, the path is the same directory as the script was run.
        The file name is prefix + suffix. By default, the file name is
        the current string stored in params['prefix']. If not specified,
        return a warning that the file name must be specified.
        '''

        path = self.params['save_path']
        if prefix is None:
            if self.params['save_prefix'] is None:
                print(('No filename was given. '
                    '\nSpecify a filename with the prefix and suffix arguments. '
                    '\nAlternatively, specify the save_prefix class instance '
                    'attribute before calling this method.'))
                return
            prefix = self.params['save_prefix']
        
        current_attributes = {
            'training_history': self.continuous_training_history,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
            }
        
        save_string = os.path.join(path, prefix)
        if suffix != '':
            save_string += '_' + suffix
        torch.save(current_attributes, save_string + '.pt')

    def load_training_history(self, instance_attribute_dict):
        '''
        Load model weights, optimizer and continuous training history 
        from an instance attribute dictionary (with the same keys as 
        the one saved in the method save_attributes_to_file()

        Overwrites the current continuous_training_history attribute.
        '''
        loaded_weights = instance_attribute_dict['model']
        loaded_optimizer = instance_attribute_dict['optimizer']
        loaded_training_history = instance_attribute_dict['training_history']
        self.load_weights(loaded_weights)
        self.load_optimizer(loaded_optimizer)
        self.continuous_training_history = loaded_training_history

    # Load best weights from training
    def load_best_weights(self):
        try:
            self.model.load_state_dict(self.best_weights)
        except AttributeError:
            print(("The model has not yet been trained "
                "-- no best weights to load"))
    
    # Load weights from an input state dict
    def load_weights(self, state_dict):
        self.model.load_state_dict(state_dict)

    # Load optimizer from an input state dict 
    def load_optimizer(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    # Set learning rate, print previous value
    def set_learning_rate(self, lr):
        print(("Changing learning rate from "
        f"{self.optimizer.param_groups[0]['lr']} to {lr}"))
        self.optimizer.param_groups[0]['lr'] = lr

    # Initialize a code object as an instance attribute so that it can be reused
    def initialize_code(self, code_size):
        self.code = RotatedCode(code_size)

    def save_model(self, filename):
        try:
            torch.save(self.best_weights, filename)
        except:
            print('No weights found - Model has not yet been trained.')

    def save_scores(self, filename):
        '''
        Save currently stored training and validation accuracies
        as numpy arrays.
        The loaded file behaves as a dictionary, with keys
        'train' and 'valid' for training and validation accuracies, 
        respectively.
        '''
        np.savez(filename, 
            train_acc = np.array(self.train_accuracies),
            valid_acc = np.array(self.valid_accuracies),
            train_loss = np.array(self.train_losses),
            valid_loss = np.array(self.valid_losses)
        )

    ##########################################################
    ###########  Method for data buffer training  ############
    ##########################################################

    def train_with_data_buffer(self, 
            code_size, 
            error_rate, 
            train = False,
            save_to_file = False,
            save_file_prefix = None,
            num_iterations = 1,
            batch_size = 200, 
            buffer_size = 100,
            replacements_per_iteration = 1,
            test_size = 10000,
            criterion = None, 
            learning_rate = None,
            benchmark = False,
            learning_scheduler = False,
            validation = False,
            append = False
        ):
        '''        
        Train the decoder by generating a buffer of random syndrome graphs,
        and continuously train the network with random selections of data 
        batches from the buffer as training data.
        The true equivalence classes of the underlying errors are used
        as training labels.
        
        After each iteration, a number of batches in the buffer are replaced
        by randomly sampling new graphs.

        The input arguments 
            replacements_per_iteration
            draws_per_iteration
            batches_per_draw 
        determine how much data is taken from the buffer for training, and 
        how much new data is put into the buffer with every iteration.
        
        '''

        ##########################################################
        ######################### Setup  #########################
        ##########################################################

        if benchmark:
            time_sample = 0.
            time_fit = 0.
            time_setup_start = time.perf_counter()

        if save_to_file:
            if save_file_prefix is None:
                if self.params['save_prefix'] is None:
                    print(('No filename was given.'
                    '\nSpecify a filename with the prefix and suffix arguments. '
                    '\nAlternatively, specify the save_prefix class instance '
                    'attribute before calling this method.'))
                    return

        params = self.params
        model = self.model
        optimizer = self.optimizer


        if self.code is None:
            self.initialize_code(code_size)
        code = self.code
        H_check = code.H

        # If learning rate is not specified, use current learning rate
        # of optimizer (0.01 if initialized by default).
        if learning_rate is None:
            learning_rate = optimizer.param_groups[0]['lr'] 
        optimizer.param_groups[0]['lr'] = learning_rate

        # Get graph structure variables from parameter dictionary
        num_node_features = params['graph']['num_node_features']
        power = params['graph']['power']
        cuda = params['cuda']
        m_nearest_nodes = params['graph']['m_nearest_nodes']

        criterionX = torch.nn.BCEWithLogitsLoss()
        criterionZ = torch.nn.BCEWithLogitsLoss()

        sigmoid = torch.nn.Sigmoid() # To convert binary network output to class index
        if cuda:
            # Use GPU if available
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else: 
            device = 'cpu'

        ####################################################################
        # DEFINE HELPER FUNCTIONS
        ####################################################################

        def generate_buffer():
            args = (batch_size, H_check, code_size, error_rate, m_nearest_nodes, power, num_node_features)
            repeated_args = [args] * buffer_size
            # create batches in parallel:
            with Pool(processes = (cpu_count() - 1)) as pool:
                buffer = pool.starmap(generate_batch, repeated_args)
            # flatten the buffer:
            buffer = [item for sublist in buffer for item in sublist]
            torch_buffer = []
            # convert list of numpy arrays to torch Data object containing torch GPU tensors
            for i in range(len(buffer)):
                X = torch.from_numpy(buffer[i][0]).to(device)
                edge_index = torch.from_numpy(buffer[i][1]).to(device)
                edge_attr = torch.from_numpy(buffer[i][2]).to(device)
                y = torch.from_numpy(buffer[i][3]).to(device)
                torch_buffer.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y))
            del buffer
            return torch_buffer

        def update_buffer(buffer, append):
            if not append:
                # delete the first entries of the buffer:
                del buffer[: (replacements_per_iteration * batch_size * len(error_rate))]
            # create a list of repeated arguments for all processes: 
            args = (batch_size, H_check, code_size, error_rate, m_nearest_nodes, power, num_node_features)
            repeated_args = [args] * replacements_per_iteration
            # create batches in parallel:
            with Pool(processes = (cpu_count() - 1)) as pool:
                new_data = pool.starmap(generate_batch, repeated_args)
            # flatten the data: 
            new_data = [item for sublist in new_data for item in sublist]
            torch_data = []
            # convert list of numpy arrays to torch Data object containing torch GPU tensors
            for i in range(len(new_data)):
                X = torch.from_numpy(new_data[i][0]).to(device)
                edge_index = torch.from_numpy(new_data[i][1]).to(device)
                edge_attr = torch.from_numpy(new_data[i][2]).to(device)
                y = torch.from_numpy(new_data[i][3]).to(device)
                torch_data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y))
            # append to buffer:
            buffer.extend(torch_data)
            del new_data, torch_data
            return buffer

        def train_with_buffer(graph_list, shuffle=True):
            '''Trains the network with data from the buffer.'''
            loader = DataLoader(graph_list, batch_size=batch_size, shuffle=shuffle)
            total_loss = 0.
            correct_predictions = 0
            model.train()
            # tensor sizes: 
            # data.x            (number of nodes in sample * batch_size, 4)
            # data.edge_index   (2, number of edge_indices in sample * batch_size)
            # data.edge_attr    (number of edge_indices in sample * batch_size, 1)
            # data.y            (batch_size, 2 for two-head 4 for one-head)
            for data in loader:
                optimizer.zero_grad()
                data.batch = data.batch.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                target = data.y.to(int)
                lossX = criterionX(out[:,0], data.y[:,0])
                lossZ = criterionZ(out[:,1], data.y[:,1])
                loss = lossX + lossZ
                prediction = (sigmoid(out.detach()) > 0.5).to(device).long()
                correct_predictions += int(((prediction == target).sum(dim=1) == 2).sum().item())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * data.num_graphs

            return correct_predictions, total_loss
        
        def generate_test_batch(test_size):
            if not train:
                args = (batch_size, H_check, code_size, error_rate, m_nearest_nodes, power, num_node_features)
                repeated_args = [args] * test_size
                # create batches in parallel:
                with Pool(processes = (cpu_count() - 1)) as pool:
                    results = pool.starmap(skapa_test_batch, repeated_args)
                buffer, correct_predictions_trivial = zip(*results)
                # flatten the buffer:
                buffer = [item for sublist in buffer for item in sublist]
                correct_predictions_trivial = np.array(list(correct_predictions_trivial)).sum()
                torch_buffer = []
                # convert list of numpy arrays to torch Data object containing torch GPU tensors
                for i in range(len(buffer)):
                    X = torch.from_numpy(buffer[i][0]).to(device)
                    edge_index = torch.from_numpy(buffer[i][1]).to(device)
                    edge_attr = torch.from_numpy(buffer[i][2]).to(device)
                    y = torch.from_numpy(buffer[i][3]).to(device)
                    torch_buffer.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y))
                del buffer
            return torch_buffer, correct_predictions_trivial

        def count_correct_predictions_in_test_batch(graph_batch):
            '''Counts the correct predictions by the network for a test batch'''
            loader = DataLoader(graph_batch, batch_size = 1000)
            correct_predictions = 0
            model.eval()            # run network in training mode 
            with torch.no_grad():   # turn off gradient computation (https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615)
                for data in loader:
                    # Perform forward pass to get network output
                    data.batch = data.batch.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    target = data.y.to(int) # Assumes binary targets (no probabilities)
                    # Sum correct predictions
                    prediction = sigmoid(out.detach()).round().to(int)
                    correct_predictions += int( ((prediction == target).sum(axis=1) == 2).sum() )

            return correct_predictions
        
        def decode_test_batch(graph_batch):
            '''
            Estimates the decoder's logical success rate with one batch 
            of test graphs which were generated in advance. Returns both the accuracy
            tested with trivial syndromes ('test_accuracy') and the accuracy tested 
            without trivial syndromes ('val_accuracy')
            '''
            # Count correct predictions by GNN for nontrivial syndromes
            correct_predictions_nontrivial = count_correct_predictions_in_test_batch(graph_batch)
            val_accuracy = correct_predictions_nontrivial / len(graph_batch)

            return val_accuracy

        def generate_and_decode_test_batch(test_size):
            '''
            Estimates the decoder's logical success rate with one batch 
            of test graphs which are generated with this function
            '''
            trivial = 0
            nontrivial = 0
            # Generate a test batch
            graph_batch, correct_predictions_trivial = generate_test_batch(test_size)
            # Count correct predictions by GNN for nontrivial syndromes
            correct_predictions_nontrivial = count_correct_predictions_in_test_batch(graph_batch)
            trivial += correct_predictions_trivial
            nontrivial += correct_predictions_nontrivial
            print(trivial, nontrivial)
            test_accuracy = (nontrivial + trivial) / (test_size * batch_size)
            return test_accuracy

        ##############################################################################
        ############################ TESTING (default)################################
        ##############################################################################

        if not train:
            test_accuracy = generate_and_decode_test_batch(test_size)
            return test_accuracy

        ##############################################################################
        ################################ TRAINING ####################################
        ##############################################################################
        if save_to_file:
            print('Will save final results to file after training.')
        if append:
            append_string = 'appending'
        else:
            append_string = 'replacing'
        print(f'Generating data with {cpu_count()} CPU cores, then moving it to device {device}.')
        print((f'Starting training with {num_iterations} iteration(s).'
            f'\nBuffer has {buffer_size * batch_size * len(error_rate)} samples, {append_string} {replacements_per_iteration * batch_size * len(error_rate)} samples with each iteration.'
            f'\nTotal number of unique samples in this run: {len(error_rate)*batch_size*(buffer_size+num_iterations*replacements_per_iteration):.2e}'))
        previously_completed_samples = self.continuous_training_history['num_samples_trained']
        if previously_completed_samples > 0:
            print(f'Cumulative # of training samples from previous runs: {previously_completed_samples:.2e}')

        # Store training parameters in history instance attribute
        self.continuous_training_history['batch_size'] = batch_size
        self.continuous_training_history['buffer_size'] = buffer_size
        self.continuous_training_history['replacements_per_iteration'] = replacements_per_iteration
        self.continuous_training_history['code_size'] = code_size
        self.continuous_training_history['training_error_rate'] = error_rate
        self.continuous_training_history['learning_rate'] = learning_rate


        # time for training setup:
        if benchmark:
            time_setup_end = time.perf_counter()
            sample_start = time.perf_counter()
    
        # Initialize data buffer
        data_buffer = generate_buffer()
        gc.collect()

        # Initialize list of validation accuracies = if it does not yet exist and generate test and validation batch
        if validation:
            test_val_batch, correct_predictions_trivial = generate_test_batch(test_size)
            try:
                self.continuous_training_history['val_acc'] == []
            except KeyError:
                self.continuous_training_history['val_acc'] = []
        
        # time for sampling:
        if benchmark:
            sample_end = time.perf_counter()
            time_sample += (sample_end - sample_start)
        
        
        # If >100 iterations, limit number of progress prints to 100
        if num_iterations > 100: 
            training_print_spacing = int(num_iterations/100)

        # Learning rate scheduler:
        if learning_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
            factor=0.5, patience=2, threshold=0.0001)

        for ix in range(num_iterations):
            if benchmark:
                fit_start = time.perf_counter()
            # forward pass:
            correct_count, total_iteration_loss = train_with_buffer(data_buffer)
            sample_count = len(data_buffer)
            # Benchmarking
            if benchmark:
                fit_end = time.perf_counter()
                time_fit += (fit_end - fit_start)
                sample_start = time.perf_counter()
            # update the data buffer (either replace or append batches)
            if replacements_per_iteration > 0:
                data_buffer = update_buffer(data_buffer, append)
            gc.collect()
            
            # Store loss and accuracy from training iteration
            average_sample_loss = total_iteration_loss / sample_count
            iteration_accuracy = correct_count / sample_count

            # Benchmarking
            if benchmark:
                sample_end = time.perf_counter()
                time_sample += (sample_end - sample_start)

            # If validation, test on validation batch
            if validation:
                val_acc = decode_test_batch(test_val_batch)
                self.continuous_training_history['val_acc'].append(val_acc)
            # save training iteration metrics
            self.continuous_training_history['loss'].append(average_sample_loss)
            self.continuous_training_history['accuracy'].append(iteration_accuracy)
            self.continuous_training_history['num_samples_trained'] += sample_count
            # Print training results. If running >100 iterations, increase spacing of prints
            if (num_iterations <= 100) or ((ix+1) % training_print_spacing == 0) or (ix == 0):
                training_print_string = (f'Iteration: {(ix+1):03d}\t'
                                        f'Loss: {average_sample_loss:.4f}\t'
                                        f'Acc: {iteration_accuracy:.4f}\t')
                if validation:
                    training_print_string += f'Validation Acc: {val_acc:.4f}\t'
                training_print_string += (f'Cumulative # of training samples: '
                                        f'{(previously_completed_samples + sample_count * (ix+1)):.2e}')
                print(training_print_string)

        print('Completed all training iterations!')
        if save_to_file:
            print('Saving final model and history to file.')
            self.save_attributes_to_file(
                prefix = save_file_prefix, 
                suffix = '')

        # BENCHMARKING
        if benchmark:
            print('\n==== BENCHMARKING ====')
            time_setup = time_setup_end - time_setup_start
            print(f'Training setup: {time_setup}')
            print(f'Sampling and Graphing: {time_sample}')
            print(f'Fitting: {time_fit}')
            print(f'\tSum: {time_setup + time_sample + time_fit}')
        
        return

def generate_batch(batch_size, H_check, code_size, error_rate, m_nearest_nodes, power, num_node_features):
    batch = []
    # need to create a different seed in every thread: 
    # https://stackoverflow.com/questions/12915177/same-output-in-different-workers-in-multiprocessing
    np.random.seed()
    for _ in range(batch_size):
        for p in error_rate:
            count = 0
            while count < 1:
                graph = generate_sample_2D(H_check, code_size, p, m_nearest_nodes, power, num_node_features)
                if graph == None:
                    continue
                count += 1
                batch.append(graph)
    return batch

def skapa_test_batch(batch_size, H_check, code_size, error_rate, m_nearest_nodes, power, num_node_features):
    batch = []
    correct_predictions_trivial = 0
    # need to create a different seed in every thread: 
    # https://stackoverflow.com/questions/12915177/same-output-in-different-workers-in-multiprocessing
    np.random.seed()
    for _ in range(batch_size):
        graph = generate_sample_2D(H_check, code_size, error_rate, m_nearest_nodes, power, num_node_features)
        if graph == None:
            correct_predictions_trivial += 1
            continue
        batch.append(graph)
    return batch, correct_predictions_trivial


def generate_sample_2D(H_check, code_size, p, m_nearest_nodes, power, num_node_features):
    # Randomly sample error chain
    error_chain = generate_error_chain(p, code_size)
    # Determine the syndrome consistent with the error
    syndrome = get_syndrome(error_chain, H_check, code_size)
    if syndrome.sum() == 0:
        return
    # Determine the equivalence class of the error chain
    true_eq_class = get_eq_class(error_chain, code_size)
    # Produce the syndrome graph
    # [X, edge_index, edge_attr, y]:
    graph = get_torch_graph(
        syndrome = syndrome,
        target = true_eq_class,
        num_node_features = num_node_features,
        power = power,
        m_nearest_nodes = m_nearest_nodes)
    return graph

def generate_error_chain(p, code_size):
    n_qubits = int(code_size ** 2)
    p_x = p_y = p_z = p/3 # Unbiased noise
    noise = np.zeros(n_qubits, dtype = np.uint8)
    draws = np.random.random(n_qubits)
    noise[(draws <= p_x)] = 1
    noise[(p_x < draws) & (draws <= p_x+p_y)] = 2
    noise[(p_x+p_y < draws) & (draws <= p_x+p_y+p_z)] = 3
    return noise

def get_syndrome(error_string, H_check, code_size):
    # Create separate binary strings for X and Z errors
    error_string_x = (error_string == 1) + (error_string == 2)
    error_string_z = (error_string == 3) + (error_string == 2)

    # Re-order X-errors to get correct qubit indices for Z-stabilizers
    error_string_x = np.rot90(error_string_x.reshape(code_size, code_size)).flatten()

    # Since they are identical, use same check matrix for both Z and X
    # Create separate syndrome strings for X and Z stabilizers
    # Note: X errors are checked by Z stabilizers and vice versa
    syndrome_string_x = H_check@error_string_z % 2
    syndrome_string_z = (H_check@error_string_x % 2) * 3
    
    # convert syndrome strings to matrix:
    M = code_size + 1
    syndrome_matrix_X = np.zeros((M, M), dtype=np.uint8)
    syndrome_string_x = syndrome_string_x.reshape(M, - 1)
    syndrome_matrix_X[::2, 2::2] = syndrome_string_x[::2]
    syndrome_matrix_X[1::2, 1:M - 2:2] = syndrome_string_x[1::2]

    syndrome_matrix_Z = np.zeros((M, M), dtype=np.uint8)
    syndrome_string_z = syndrome_string_z.reshape(M, - 1)
    syndrome_matrix_Z[::2, 2::2] = syndrome_string_z[::2]
    syndrome_matrix_Z[1::2, 1:M - 2:2] = syndrome_string_z[1::2]
    # Combine syndrome matrices where 1 entries 
    # correspond to x and 3 entries to z defects
    syndrome_matrix = syndrome_matrix_X + np.rot90(syndrome_matrix_Z, - 1)
    
    # Return the syndrome matrix
    return syndrome_matrix

def get_eq_class(error, code_size):
    error_matrix = error.reshape(code_size, code_size)
    # Count number of Z-errors on west edge
    Z_count = np.count_nonzero( 
        (error_matrix[:,0] == 3) + (error_matrix[:,0] == 2) ) 
    # Count number of X-errors on north edge
    X_count = np.count_nonzero(
        (error_matrix[0,:] == 1) + (error_matrix[0,:] == 2) ) 
    # Determine equivalence class from parity of Z-errors on west edge
    # and X-errors on north edge
    if Z_count % 2 == 0:
        if X_count % 2 == 0:
            return np.array([0, 0]) # Equivalence class I
        else:
            return np.array([1, 0]) # Equivalence class X
    else:
        if X_count % 2 == 0:
            return np.array([0, 1]) # Equivalence class Z
        else:
            return np.array([1, 1]) # Equivalence class Y