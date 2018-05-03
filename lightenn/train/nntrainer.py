import numpy as np
from lightenn import utils
from lightenn import types

class NNTrainer:
    
    def __init__(self, layers, config):
        
        self.layers = layers
        self.config = config
    
    # To support Stochastic Gradient Descent (SGD):
    #
    # For each epoch:
    #
    # - Shuffle training data
    # - Sample individual examples without replacement
    # - For each example, compute your gradient vector and adjust the weights
    def train_sgd(self, training_set, num_epochs, validation_set=None):
    
        # Start by doing some basic checks on the inputs:
        self._check_training_set(training_set)
        training_x = training_set[0]
        training_y = training_set[1]

        if validation_set is not None:
            self._check_training_set(validation_set)

        # Run num_epochs of training:
        for epoch in range(num_epochs):
        
            print('Starting epoch', epoch+1)
            
            epoch_error = 0.0
            
            # Shuffle your data. Adapted from:
            # https://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
            indices = np.arange(len(training_x))
            np.random.shuffle(indices)
            training_x_shuffled = training_x[indices]
            training_y_shuffled = training_y[indices]
            
            for i, x in enumerate(training_x_shuffled):
                
                # Load dropout masks for this example, if applicable
                self._load_dropout_masks()
                
                # Train on the example
                y = training_y_shuffled[i]
                self.layers[0].values = x # feed the input values
                self._forward() # do the forward pass
                y_hat = self.layers[len(self.layers)-1].activations # get y_hat
                epoch_error += utils.compute_example_loss(y, y_hat, self.layers,
                                                          self.config['loss_type'],
                                                          self.config['regularizer']) # compute error for this example
                self._backward_compute_grads(y, y_hat)
                
                # Adjust the weights and biases
                self._backward_adjust()
                
                # Clear dropout masks, if applicable
                self._clear_dropout_masks()
            
            # Output error for this epoch
            print('Error after epoch', epoch+1, ':', epoch_error)
            
            # Run validation after every epoch, if validation set supplied
            if validation_set is not None:
                val_acc = self.get_accuracy(validation_set)
                print('Validation accuracy after epoch', epoch+1, ':', val_acc)

    # If y_hat has size 1, we threshold our model outputs to 0 or 1 at test time.
    def get_accuracy(self, data_set):
        
        self._check_training_set(data_set)
        training_x = data_set[0]
        training_y = data_set[1]
        num_correct = 0
        
        for x_i, x in enumerate(training_x):
            y = training_y[x_i]
            y_hat = self.predict(x)
            if len(y_hat) > 1:
                if np.argmax(y_hat) == np.argmax(y):
                    num_correct += 1
            else:
                prediction = None
                y_hat = y_hat[0]
                if y_hat >= 0.5:
                    prediction = 1.0
                else:
                    prediction = 0.0
                if prediction == y:
                    num_correct += 1

        return round(1.0 * num_correct / len(training_x), 2)

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

    def predict(self, x):
        self.layers[0].values = x # feed the input values
        self._forward() # do the forward pass
        y_hat = self.layers[len(self.layers)-1].activations # get y_hat
        return y_hat

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

