from lightenn import types
from lightenn.build import nnbuilder
from lightenn.grads import nngradchecker
from lightenn.train import nntrainer

# Default learning rate
LEARNING_RATE=0.001

# Default mu and stddev when generating random weights and biases
STDDEV=1.0
MU=0.0

# Default delta and epsilon for numerical gradient checking
DELTA=0.0001
EPSILON=1e-8

ERR_NOT_INIT = 'Error: neural net has not been initialized.'

class NeuralNet:

    def __init__(self):
        
        self.config = {}
        self.initialized = False
        self.layers = []
        
        self.builder = nnbuilder.NNBuilder(self.layers, self.config)
        self.gradchecker = nngradchecker.NNGradChecker(self.layers, self.config)
        self.trainer = nntrainer.NNTrainer(self.layers, self.config)
    
    def add_input(self, size, values=None, dropout_p=0.0):
        self.builder.add_input(size, values=values, dropout_p=dropout_p)
    
    def add_fully_connected(self, size, activation_type=types.ActivationType.SIGMOID,
                            wgts=None, biases=None, dropout_p=0.0):
        self.builder.add_fully_connected(size, activation_type=activation_type,
                                         wgts=wgts, biases=biases, dropout_p=dropout_p)

    def add_output(self, size, activation_type=types.ActivationType.SIGMOID, wgts=None, biases=None):
        self.builder.add_output(size, activation_type=activation_type, wgts=wgts, biases=biases)

    def initialize(self, loss_type=types.LossType.SQUARED_ERROR,
                   learning_rate=LEARNING_RATE, mu=MU, stddev=STDDEV,
                   regularizer=None):
        
        self.config['loss_type'] = loss_type
        self.config['learning_rate'] = learning_rate
        self.config['mu'] = mu
        self.config['stddev'] = stddev
        self.config['regularizer'] = regularizer
        
        self.builder.initialize(self.config)
        self.initialized = True
    
    def train_sgd(self, training_set, num_epochs, validation_set=None):
        assert (self.initialized), ERR_NOT_INIT
        self.trainer.train_sgd(training_set, num_epochs, validation_set=validation_set)

    def get_accuracy(self, data_set):
        assert (self.initialized), ERR_NOT_INIT
        return self.trainer.get_accuracy(data_set)
    
    def predict(self, x):
        assert (self.initialized), ERR_NOT_INIT
        return self.trainer.predict(x)

    # TODO: A full numerical gradient check on a large network can take a very
    # long time. Add a 'stochastic' parameter.
    def check_gradients(self, y, delta=DELTA, epsilon=EPSILON):
        assert (self.initialized), ERR_NOT_INIT
        self.gradchecker.check_gradients(y, delta, epsilon)



