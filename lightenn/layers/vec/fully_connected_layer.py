import numpy as np
from lightenn import types
from lightenn import utils
from lightenn.layers.vec import base_layer

class FullyConnectedLayer(base_layer.BaseLayer):

    def __init__(self, nn_config, size, idx,
                 activation_type=types.ActivationType.SIGMOID,
                 wgts=None, biases=None, dropout_p=0.0):

        super().__init__(nn_config, size, idx)
        self.activations = None
        self.size = size
        self.idx = idx
        self.activation_type = activation_type
        self.wgts = wgts
        self.biases = biases
        self.z_vals = None
        self.cost_wrt_activations = []  # partial derivatives of cost w.r.t. my activations
        self.cost_wrt_wgts = []  # partial derivatives of cost w.r.t. weights
        self.cost_wrt_biases = []  # partial derivatives of cost w.r.t. biases
        self.dropout_p = dropout_p
        self.dropout_mask = np.ones((size,), dtype=np.float)

    def initialize(self):

        assert (self.size > 0), 'Error: cannot add a layer of size 0.'
        assert (self.dropout_p >= 0.0 and self.dropout_p < 1.0), 'Error: dropout_p must be in the range [0.0, 1.0).'

        if self.wgts is None:
            self.wgts = np.random.normal(loc=self.nn_config['mu'],
                                         scale=self.nn_config['stddev'],
                                         size=(self.prev.size, self.size))
        else:
            assert (len(self.wgts.shape) == 2), 'Error: wgts ndarray must be 2-D.'
            assert (self.wgts.dtype == np.float), 'Error: wgts must have dtype np.float.'
            assert (self.wgts.shape[0] == self.prev.size), 'Error: wgts.shape[0] must match previous layer size.'
            assert (self.wgts.shape[1] == self.size), 'Error: wgts.shape[1] must match layer size.'

        if self.biases is None:
            self.biases = np.random.normal(loc=self.nn_config['mu'],
                                           scale=self.nn_config['stddev'],
                                           size=(self.size,))
        else:
            assert (len(self.biases.shape) == 1), 'Error: biases ndarray must be 1-D.'
            assert (self.biases.dtype == np.float), 'Error: biases must have dtype np.float.'
            assert (self.biases.shape[0] == self.size), 'Error: size of biases ndarray must match layer size.'

    def activate(self, z_vals):
        if self.activation_type == types.ActivationType.SIGMOID:
            return utils.sigmoid(z_vals)
        elif self.activation_type == types.ActivationType.RELU:
            return utils.relu(z_vals)
        elif self.activation_type == types.ActivationType.NONE:
            return z_vals

    def activation_prime(self, z_vals, activation_type):
        if activation_type == types.ActivationType.SIGMOID:
            return utils.sigmoid_prime(z_vals)
        elif activation_type == types.ActivationType.RELU:
            return utils.relu_prime(z_vals)
        elif activation_type == types.ActivationType.NONE:
            if isinstance(z_vals, np.float):
                return 1.0
            else:
                return np.ones_like(z_vals)

    # Computes partial regularization terms for all incoming weights in one
    # operation
    def compute_partial_reg_term(self):

        batch_size = self.activations.shape[0]
        reg = self.nn_config['regularizer']

        if reg == None:
            return np.zeros_like(self.wgts)

        reg_type = reg[0]
        lambd = reg[1]

        if reg_type == types.RegType.L1:
            # reg_terms[wgts == 0] are already 0.0 since init'd as zero-array.
            reg_terms = np.zeros_like(self.wgts)
            reg_terms[self.wgts > 0.0] = lambd / batch_size
            reg_terms[self.wgts < 0.0] = -lambd / batch_size
            return reg_terms

        elif reg_type == types.RegType.L2:
            return np.multiply(lambd / batch_size, self.wgts)

    def compute_cost_wrt_activations(self, batch_size, y, y_hat):

        next_zs_prime = self.activation_prime(self.next.z_vals, self.next.activation_type)
        a = next_zs_prime
        b = self.next.wgts.reshape(self.next.wgts.shape[0], 1, self.next.wgts.shape[1])
        next_act_wrt_curr_act = np.multiply(a, b)
        cost_wrt_next_acts = self.next.cost_wrt_activations
        cost_wrt_acts = np.sum(np.multiply(cost_wrt_next_acts, next_act_wrt_curr_act), axis=2).transpose()

        # Dropout: gradient of cost w.r.t. my activations must
        # also now include the dropout mask and inverse terms that are
        # associated with my layer
        return np.divide(np.multiply(cost_wrt_acts, self.dropout_mask), 1.0 - self.dropout_p)

    def forward(self):

        self.z_vals = np.add(np.dot(self.prev.activations, self.wgts), self.biases)
        self.activations = self.activate(self.z_vals)

        # Dropout: in the forward pass, set to 0 any activations that are masked
        # by the dropout_mask. Apply the invert as well with divide.
        self.activations = np.divide(np.multiply(self.activations, self.dropout_mask), 1.0 - self.dropout_p)

    def backward_compute_grads(self, y, y_hat):

        # Batch size that was fed in at the input layer. Because the dot product
        # is used for every forward, this will always be the value of the first
        # dimension of the activation layer
        batch_size = self.activations.shape[0]

        # Compute gradient of my activations wrt wgts and biases
        activations_prime = self.activation_prime(self.z_vals, self.activation_type)
        in_acts = self.prev.activations
        a = in_acts.reshape(batch_size, in_acts.shape[1], 1)
        b = activations_prime.reshape(batch_size, 1, activations_prime.shape[1])
        # acts_wrt_wgts = np.einsum('...hij,...hjk->...hik', a, b)
        acts_wrt_wgts = np.multiply(a, b)
        acts_wrt_bias = activations_prime

        # compute grad of cost wrt my activations
        self.cost_wrt_activations = self.compute_cost_wrt_activations(batch_size, y, y_hat)

        # compute cost w.r.t. weights and biases
        a = acts_wrt_wgts
        b = self.cost_wrt_activations.reshape(
            (self.cost_wrt_activations.shape[0], 1, self.cost_wrt_activations.shape[1]))
        self.cost_wrt_wgts = np.sum(np.multiply(a, b), axis=0)
        self.cost_wrt_biases = np.sum(np.multiply(self.cost_wrt_activations, acts_wrt_bias), axis=0)

        # Add partial regularization term, as applicable
        partial_reg_term = self.compute_partial_reg_term()
        self.cost_wrt_wgts = np.add(self.cost_wrt_wgts, partial_reg_term)

    def backward_adjust(self):

        learning_rate = self.nn_config['learning_rate']
        self.wgts = np.add(self.wgts, np.multiply(-1.0 * learning_rate, self.cost_wrt_wgts))
        self.biases = np.add(self.biases, np.multiply(-1.0 * learning_rate, self.cost_wrt_biases))
