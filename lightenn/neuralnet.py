import numpy as np
from typing import Tuple
from lightenn import types
from lightenn.build import nnbuilder
from lightenn.grads import nngradchecker
from lightenn.train import nntrainer

# Default learning rate
LEARNING_RATE = 0.001

# Default mu and stddev when generating random weights and biases
STDDEV = 1.0
MU = 0.0

# Default delta and epsilon for numerical gradient checking
DELTA = 0.0001
EPSILON = 1e-8

ERR_NOT_INIT = 'Error: neural net has not been initialized.'
ERR_ALREADY_INIT = 'Error: neural net has already been initialized.'
ERR_NO_ADD = 'Error: cannot add more layers after initialization.'

# The neuralnet class exposes the main API for building, initializing, training, and running predictions from networks.
# We confine our static type-checks to this class, in hopes that eventually Python will fully support them.
class NeuralNet:

    def __init__(self):

        self.config = {}
        self.initialized = False
        self.layers = []

        self.builder = nnbuilder.NNBuilder(self.layers, self.config)
        self.gradchecker = nngradchecker.NNGradChecker(self.layers, self.config)
        self.trainer = nntrainer.NNTrainer(self.layers, self.config)

    def add_input(self,
                  size: int,
                  values: np.ndarray = None,
                  dropout_p: float = 0.0):

        assert (not self.initialized), ERR_NO_ADD
        self.builder.add_input(size, values=values, dropout_p=dropout_p)

    def add_fully_connected(self,
                            size: int,
                            activation_type: types.ActivationType = types.ActivationType.SIGMOID,
                            wgts: np.ndarray = None,
                            biases: np.ndarray = None,
                            dropout_p: float = 0.0):

        assert (not self.initialized), ERR_NO_ADD
        self.builder.add_fully_connected(size, activation_type=activation_type,
                                         wgts=wgts, biases=biases, dropout_p=dropout_p)

    def add_output(self,
                   size: int,
                   activation_type: types.ActivationType = types.ActivationType.SIGMOID,
                   wgts: np.ndarray = None,
                   biases: np.ndarray = None):

        assert (not self.initialized), ERR_NO_ADD
        self.builder.add_output(size, activation_type=activation_type, wgts=wgts, biases=biases)

    def initialize(self,
                   loss_type: types.LossType = types.LossType.SQUARED_ERROR,
                   learning_rate: float = LEARNING_RATE,
                   mu: float = MU,
                   stddev: float = STDDEV,
                   regularizer: types.RegType = None,
                   target: types.TargetType = types.TargetType.CPU):

        assert (not self.initialized), ERR_ALREADY_INIT

        self.config['loss_type'] = loss_type
        self.config['learning_rate'] = learning_rate
        self.config['mu'] = mu
        self.config['stddev'] = stddev
        self.config['regularizer'] = regularizer
        self.config['target'] = target

        self.builder.initialize()
        self.initialized = True

    # To do a full-batch train, we simply do a mini-batch train with a
    # batch_size = m.
    def train_full(self,
                   training_set: Tuple[np.ndarray, np.ndarray],
                   num_epochs: int,
                   shuffle: bool = True,
                   validation_set: Tuple[np.ndarray, np.ndarray] = None,
                   verbose: bool = False):

        assert (self.initialized), ERR_NOT_INIT
        m = training_set[0].shape[0]
        self.trainer.train_batch(training_set, num_epochs, batch_size=m,
                                 shuffle=shuffle, validation_set=validation_set, verbose=verbose)

    # SGD should select single training examples at random, computing gradients
    # and updating weights and biases for each, until exhaustion. This is
    # equivalent to calling train_batch() with a batch_size of 1 and the shuffle
    # flag set to True.
    def train_sgd(self,
                  training_set: Tuple[np.ndarray, np.ndarray],
                  num_epochs: int,
                  validation_set: Tuple[np.ndarray, np.ndarray] = None,
                  verbose: bool = False):

        assert (self.initialized), ERR_NOT_INIT
        self.trainer.train_batch(training_set,
                                 num_epochs,
                                 batch_size=1,
                                 shuffle=True,
                                 validation_set=validation_set,
                                 verbose=verbose)

    def train_mini_batch(self,
                         training_set: Tuple[np.ndarray, np.ndarray],
                         num_epochs: int,
                         batch_size: int = 64,
                         shuffle: bool = True,
                         validation_set: Tuple[np.ndarray, np.ndarray] = None,
                         verbose: bool = False):

        assert (self.initialized), ERR_NOT_INIT
        self.trainer.train_batch(training_set, num_epochs, batch_size=batch_size,
                                 shuffle=shuffle, validation_set=validation_set, verbose=verbose)

    def get_accuracy(self,
                     data_set: Tuple[np.ndarray, np.ndarray],
                     batch_size: int = 64):

        assert (self.initialized), ERR_NOT_INIT
        return self.trainer.get_accuracy(data_set, batch_size=batch_size)

    def predict(self,
                input_x: np.ndarray):

        assert (self.initialized), ERR_NOT_INIT
        return self.trainer.predict(input_x)

    # TODO: A full numerical gradient check on a large network can take a very
    # long time. Add a 'stochastic' parameter.
    def check_gradients(self,
                        delta: float = DELTA,
                        epsilon: float = EPSILON,
                        y: np.ndarray = None,
                        verbose: bool = False,
                        quit_on_fail: bool = True):

        assert (self.initialized), ERR_NOT_INIT
        self.gradchecker.check_gradients(delta, epsilon, y=y, verbose=verbose, quit_on_fail=quit_on_fail)
