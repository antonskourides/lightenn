import numpy as np
from lightenn import utils

class NNTrainer:
    
    def __init__(self, layers, config):
        
        self.layers = layers
        self.config = config
    
    # This training function can be used to do each of Full-Batch, Stochastic,
    # or mini-batch Gradient Descent.
    def train_batch(self, training_set, num_epochs,
                    batch_size=64, shuffle=True, validation_set=None, verbose=False):
        
        # Check your training set and validation sets
        self._check_training_set(training_set)
        if validation_set is not None:
            self._check_training_set(validation_set)

        training_x_orig = training_set[0]
        training_y_orig = training_set[1]
        
        for e in range(num_epochs):
        
            print('Starting epoch', e+1, '...')
            
            training_x = None
            training_y = None
            
            # Shuffle your data if shuffle flag is True. Adapted from:
            # https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
            if shuffle == True:
                
                if verbose == True:
                    print('Shuffling data ...')
                
                indices = np.arange(training_x_orig.shape[0])
                np.random.shuffle(indices)
                # From https://docs.scipy.org/doc/numpy-1.13.0/user/basics.indexing.html
                # "For all cases of index arrays, what is returned is a copy of
                # the original data, not a view as one gets for slices."
                training_x = training_x_orig[indices]
                training_y = training_y_orig[indices]
            else:
                training_x = training_x_orig
                training_y = training_y_orig
        
            batch_idx = 0
            loss = 0.
        
            while batch_idx < len(training_y):
                
                if verbose == True:
                    print('Current batch index is', batch_idx, 'with training set of size', len(training_y))
                
                batch_x = training_x[batch_idx:batch_idx+batch_size]
                batch_y = training_y[batch_idx:batch_idx+batch_size]
                self.layers[0].values = batch_x
                self._forward()
                batch_y_hat = self.layers[len(self.layers)-1].activations
                self._backward_compute_grads(batch_y, batch_y_hat)
                self._backward_adjust()
                loss += utils.compute_batched_loss(batch_y, batch_y_hat,
                                                   self.layers,
                                                   self.config['loss_type'],
                                                   self.config['regularizer'])
                batch_idx += batch_size
        
            print('Total loss at end of epoch', e+1, 'is', loss)
            if validation_set is not None:
                print('Validation accuracy at end of epoch', e+1, 'is',
                      self.get_accuracy(validation_set, batch_size=batch_size))

    # Returns classification accuracy on the passed data_set. We threshold our
    # model outputs to 0 or 1 at test time when output layer size is 1.
    def get_accuracy(self, data_set, batch_size=64):
        
        # Check your data set
        self._check_training_set(data_set)
        data_x = data_set[0]
        data_y = data_set[1]
        num_correct = 0
        batch_idx = 0
        
        while batch_idx < len(data_y):

            # Python gracefully handles instances when a slice overshoots the end of an array
            batch_x = data_x[batch_idx:batch_idx+batch_size]
            batch_y = data_y[batch_idx:batch_idx+batch_size]
            batch_y_hat = self.predict(batch_x)
            
            if batch_y.shape[1] > 1:
                batch_y_hat_argmax = np.argmax(batch_y_hat, axis=1)
                batch_y_argmax = np.argmax(batch_y, axis=1)
                num_correct += np.sum(batch_y_hat_argmax == batch_y_argmax)
            elif batch_y.shape[1] == 1:
                batch_y_hat_tf = (batch_y_hat >= 0.5)
                batch_y_tf = (batch_y == 1.)
                num_correct += np.sum(batch_y_hat_tf == batch_y_tf)

            batch_idx += batch_size
        
        return round(1.0 * num_correct / data_x.shape[0], 2)

    # The predict() function returns raw output activations. Any thresholding of
    # the raw activations (say, to make a classification) must be done at the
    # point of call
    def predict(self, input_x):
        
        assert (isinstance(input_x, np.ndarray)), 'Error: input_x must be an ndarray.'
        assert (len(input_x.shape) == 2), 'Error: input_x must be 2-D.'
        assert (input_x.shape[1] == self.layers[0].size), 'Error: size of input_x elements must match input layer size.'

        self.layers[0].values = input_x
        self._forward()
        return self.layers[len(self.layers)-1].activations

    # We 'forward' the input layer too, so we can apply its dropout mask if
    # needed
    def _forward(self):
        for layer in self.layers:
            layer.forward()
    
    def _backward_compute_grads(self, y, y_hat):
        for i in range(len(self.layers)-1, 0, -1):
            self.layers[i].backward_compute_grads(y, y_hat)
    
    def _backward_adjust(self):
        for i in range(len(self.layers)-1, 0, -1):
            self.layers[i].backward_adjust()

    def _load_dropout_masks(self):
        for layer in self.layers:
            if layer.dropout_p > 0.0:
                layer.load_dropout_mask()

    def _clear_dropout_masks(self):
        for layer in self.layers:
            if layer.dropout_p > 0.0:
                layer.clear_dropout_mask()

    def _check_training_set(self, training_set):
    
        assert (isinstance(training_set, tuple) and len(training_set) == 2), 'Error: training_set must be a tuple of size 2.'
        
        training_x = training_set[0]
        training_y = training_set[1]
        
        assert (isinstance(training_x, np.ndarray) and isinstance(training_y, np.ndarray)), 'Error: training tuple elements must be ndarrays.'
        assert (len(training_x.shape) == 2  and len(training_y.shape) == 2), 'Error: training tuple elements must be 2-D, even if outputs y are length 1.'
        assert (training_x.shape[0] == training_y.shape[0]), 'Error: lengths of training_x and training_y must match.'
        assert (training_y.shape[1] == self.layers[len(self.layers)-1].size), 'Error: training_y example length does not match output layer size.'

