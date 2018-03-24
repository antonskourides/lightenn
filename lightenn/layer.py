import math
import numpy as np
from lightenn import types
from lightenn import utils

class Layer:
    
    def __init__(self, nn_config, prev_size, size, idx,
                 activation_type=types.ActivationType.SIGMOID,
                 wgts=None, biases=None, dropout_p=0.0):
        
        self.nn_config = nn_config
        self.size = size
        self.idx = idx
        self.activation_type = activation_type
        self.wgts = wgts
        self.biases = biases
        
        if self.wgts is None:
            self.wgts = np.random.normal(loc=self.nn_config['mu'],
                                         scale=self.nn_config['stddev'],
                                         size=(prev_size, size))
        if self.biases is None:
            self.biases = np.random.normal(loc=self.nn_config['mu'],
                                           scale=self.nn_config['stddev'],
                                           size=(size,))
        
        self.z_vals = np.zeros((size,), dtype=np.float)
        self.activations = np.zeros((size,), dtype=np.float)
        self.grads_wgts = np.zeros((prev_size, size), dtype=np.float)
        self.grads_biases = np.zeros((size,), dtype=np.float)
        self.acc_grads = np.zeros((size,), dtype=np.float)
        self.total_grad_wgts = np.zeros((prev_size, size), dtype=np.float) # we use this in SGD
        self.total_grad_biases = np.zeros((size,), dtype=np.float) # we use this in SGD
        self.prev = None
        self.next = None
        self.dropout_p = dropout_p
        self.dropout_mask = np.ones((size,), dtype=np.float)
        self.mult_flag = False
        
    def fail_on_nan_or_inf(self, a):
        f = a.flatten()
        for i in f:
            if math.isnan(i) == True:
                exit(1)
        for i in f:
            if np.isinf(i) == True:
                exit(1)

    def compute_reg_terms(self):
        
        reg = self.nn_config['regularizer']
        lambd = reg[1]
        
        if reg[0] == types.RegType.L1:
            # reg_terms[wgts == 0] are already 0.0 since init'd as zero-array.
            reg_terms = np.zeros_like(self.wgts)
            reg_terms[self.wgts > 0.0] = lambd
            reg_terms[self.wgts < 0.0] = -lambd
            return reg_terms
        elif reg[0] == types.RegType.L2:
            return np.multiply(2.0 * lambd, self.wgts)

    def forward(self):
        
        # Do activation. Note that the activation vals in the previous layer
        # have already had Dropout mask and Inverse Dropout factor applied
        # to them
        vals = None
        if isinstance(self.prev, InputLayer):
            vals = self.prev.values
        else:
            vals = self.prev.activations
        
        wgts = self.wgts
        biases = self.biases
        self.z_vals = (np.dot(vals.reshape(1,len(vals)), wgts) + biases).reshape((self.size,))
        
        if self.activation_type == types.ActivationType.SIGMOID:
            self.activations = utils.sigmoid(self.z_vals)
        elif self.activation_type == types.ActivationType.RELU:
            self.activations = utils.relu(self.z_vals)
        elif self.activation_type == types.ActivationType.NONE:
            self.activations = self.z_vals
        
        # Dropout: in the forward pass, set to 0 any activations that are masked
        # by the dropout_mask. Apply the invert as well with divide.
        self.activations = np.divide(np.multiply(self.activations, self.dropout_mask), 1.0-self.dropout_p)
        
        # Compute first derivative of activation of the weighted sum
        activations_prime = None
        if self.activation_type == types.ActivationType.SIGMOID:
            activations_prime = utils.sigmoid_prime(self.z_vals)
        elif self.activation_type == types.ActivationType.RELU:
            activations_prime = utils.relu_prime(self.z_vals)
        elif self.activation_type == types.ActivationType.NONE:
            activations_prime = np.ones((self.size,))

        self.grads_wgts = np.dot(vals.reshape((len(vals),1)),
                                 activations_prime.reshape((1,len(activations_prime))))
        np.copyto(self.grads_biases, activations_prime)

    def backward_adjust(self):
        
        learning_rate = self.nn_config['learning_rate']
        self.wgts = np.add(self.wgts, np.multiply(learning_rate, np.multiply(-1.0, self.grads_wgts)))
        self.biases = np.add(self.biases, np.multiply(learning_rate, np.multiply(-1.0, self.grads_biases)))
    
    def clear_total_grads(self):
        self.total_grad_wgts.fill(0.0)
        self.total_grad_biases.fill(0.0)

    # Call this from the NNTrainer to provide flexibility on dropout resets
    # for both full-batch and SGD
    def load_dropout_mask(self):
        
        # There is a chance all neurons will zero out - in this case, regenerate
        non_zero = False
        while non_zero == False:
            # uniform generates over range [low, high)
            v = np.random.uniform(low=0.0, high=1.0, size=self.size)
            self.dropout_mask = (v <= self.dropout_p).astype(np.float)
            if np.sum(self.dropout_mask) > 0.0:
                non_zero = True
                
    def clear_dropout_mask(self):
        self.dropout_mask = np.ones((self.size,), dtype=np.float)

class FullyConnectedLayer(Layer):
    
    def __init__(self, nn_config, prev_size, size, idx,
                 activation_type=types.ActivationType.SIGMOID,
                 wgts=None, biases=None, dropout_p=0.0):
        super().__init__(nn_config, prev_size, size, idx,
                         activation_type=activation_type,
                         wgts=wgts, biases=biases, dropout_p=dropout_p)
        self.dropout_p = dropout_p

    def backward_compute_grads(self, y, y_hat):
        
        # Compute the regularization terms, if applicable
        reg_terms = 0.0
        if self.nn_config['regularizer'] is not None:
            reg_terms = self.compute_reg_terms()
        
        self.acc_grads = np.zeros((self.size,), dtype=np.float)
        activations_prime = None
        if self.next.activation_type == types.ActivationType.SIGMOID:
            activations_prime = utils.sigmoid_prime(self.next.z_vals)
        elif self.next.activation_type == types.ActivationType.RELU:
            activations_prime = utils.relu_prime(self.next.z_vals)
        elif self.next.activation_type == types.ActivationType.NONE:
            activations_prime = np.ones((self.next.size,))

        a = np.multiply(activations_prime, self.next.acc_grads)
        self.acc_grads = np.dot(self.next.wgts, a)
        
        # Dropout: gradient of error w.r.t. my activations (acc_grads) must
        # also now include the dropout mask and inverse terms that are
        # associated with my layer
        self.acc_grads = np.divide(np.multiply(self.acc_grads, self.dropout_mask), 1.0-self.dropout_p)
        self.grads_wgts = np.multiply(self.grads_wgts,
                                      self.acc_grads.reshape((len(self.acc_grads),1)).T) + reg_terms
        self.grads_biases = np.multiply(self.grads_biases, self.acc_grads)

class OutputLayer(Layer):
    
    def __init__(self, nn_config, prev_size, size, idx,
                 activation_type=types.ActivationType.SIGMOID,
                 wgts=None, biases=None):
        super().__init__(nn_config, prev_size, size, idx,
                         activation_type=activation_type,
                         wgts=wgts, biases=biases)

    def compute_loss_terms(self, y, y_hat):

        loss_type = self.nn_config['loss_type']

        if loss_type == types.LossType.SQUARED_ERROR:
            return -2.0 * (y - y_hat)
        elif loss_type == types.LossType.CROSS_ENTROPY:
            return -1.0 * (y/y_hat - (1.0-y)/(1.0-y_hat))

    def backward_compute_grads(self, y, y_hat):
            
        # Compute the regularization terms, if applicable
        reg_terms = 0.0
        if self.nn_config['regularizer'] is not None:
            reg_terms = self.compute_reg_terms()
            
        loss_terms = self.compute_loss_terms(y, y_hat)
        self.acc_grads = loss_terms
        self.grads_wgts = np.multiply(self.grads_wgts, loss_terms.reshape((len(loss_terms),1)).T) + reg_terms
        self.grads_biases = np.multiply(self.grads_biases, loss_terms)

class InputLayer:
    
    def __init__(self, nn_config, size, idx, values, dropout_p=0.0):
        self.nn_config = nn_config
        self.size = size
        self.idx = idx
        self.values = values
        self.dropout_p = dropout_p
        self.dropout_mask = np.ones((size,), dtype=np.float)

    # Call this from the NNTrainer to provide flexibility on dropout resets
    # for both full-batch and SGD
    def load_dropout_mask(self):
    
        # There is a chance all neurons will zero out - in this case, regenerate
        non_zero = False
        while non_zero == False:
            # uniform generates over range [low, high)
            v = np.random.uniform(low=0.0, high=1.0, size=self.size)
            self.dropout_mask = (v >= self.dropout_p).astype(np.float)
            if np.sum(self.dropout_mask) > 0.0:
                non_zero = True

    def clear_dropout_mask(self):
        self.dropout_mask = np.ones((self.size,), dtype=np.float)

    # We only 'forward' the input layer to apply its Dropout mask
    def forward(self):
        self.values = np.divide(np.multiply(self.values, self.dropout_mask), 1.0-self.dropout_p)

