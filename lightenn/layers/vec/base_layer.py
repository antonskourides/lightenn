import numpy as np

# This is the base class for all fully-connected, vector-shaped layers.
class BaseLayer:

    def __init__(self, nn_config, size, idx):

        self.nn_config = nn_config
        self.size = size
        self.idx = idx
        self.prev = None
        self.next = None

    def initialize(self):
        pass

    def load_dropout_mask(self):

        if self.dropout_p == 0.0:
            return

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
