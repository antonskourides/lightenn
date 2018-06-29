import numpy as np
from lightenn.layers.vec import base_layer

# Classic vector-shaped input layer.
class InputLayer(base_layer.BaseLayer):

    def __init__(self, nn_config, size, idx, values=None, dropout_p=0.0):

        super().__init__(nn_config, size, idx)
        self.values = values
        self.dropout_p = dropout_p
        self.dropout_mask = np.ones((self.size,), dtype=np.float)

    def check_values(self):

        assert (len(self.values.shape) == 2), 'Error: values ndarray must be 2-D: (num_examples, example_len).'
        assert (self.values.dtype == np.float), 'Error: values must have dtype np.float.'
        assert (self.values.shape[1] == self.size), 'Error: example size must match input layer size.'

    def initialize(self):

        assert (self.size > 0), 'Error: cannot add a layer of size 0.'
        assert (self.dropout_p >= 0.0 and self.dropout_p < 1.0), 'Error: dropout_p must be in the range [0.0, 1.0).'

        # check values as needed
        if self.values is None:
            self.values = np.ones((1, self.size), dtype=np.float) * 0.1
        else:
            self.check_values()

    # We only 'forward' the input layer to apply its Dropout mask
    def forward(self):

        self.check_values()
        self.activations = np.divide(np.multiply(self.values, self.dropout_mask), 1.0 - self.dropout_p)
