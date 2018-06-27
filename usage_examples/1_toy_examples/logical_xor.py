import sys
sys.path.append("../..")
import time
import numpy as np

from lightenn import neuralnet
from lightenn import types

def load_training_data(path):

    training_x = []
    training_y = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            t = line.split(',')
            training_x.append(t[0:len(t)-1])
            training_y.append(t[len(t)-1:len(t)])

    assert (len(training_x) > 0), 'Error: empty training set.'

    return (np.array(training_x, dtype=np.float),
            np.array(training_y, dtype=np.float))

def split_validation(training_x, training_y, val_perc):
    split_idx = int((1.0-val_perc)*len(training_x))
    return (training_x[0:split_idx], training_y[0:split_idx],
            training_x[split_idx:len(training_x)], training_y[split_idx:len(training_y)])

# Hyperparams
num_epochs = 10
learning_rate = 1.0

# For selecting random weights and biases
mu = 0.0
stddev = 1.0

# Percentage of validation data
val_perc = 0.2

# Load training and validation sets
training_x, training_y = load_training_data('./data/xor_data.csv')
training_x, training_y, validation_x, validation_y = split_validation(training_x, training_y, val_perc)

# Stabilize randomness
np.random.seed(123)

# Build neural net
nn = neuralnet.NeuralNet()
nn.add_input(2)
nn.add_fully_connected(3, activation_type=types.ActivationType.SIGMOID)
nn.add_output(1, activation_type=types.ActivationType.SIGMOID)
nn.initialize(loss_type=types.LossType.CROSS_ENTROPY, learning_rate=learning_rate, mu=mu, stddev=stddev)

# Numerical gradient check
nn.check_gradients(delta=0.00001)

# Train in SGD mode
t_0 = time.time()
nn.train_sgd((training_x, training_y), num_epochs, validation_set=(validation_x, validation_y))
t_1 = time.time()
tot_time = round(t_1 - t_0, 2)
print('Total time (in seconds):', tot_time)

# Predict. Remember that the output logit is a sigmoid - we must round
# to get the prediction value
print('0 XOR 0:', np.round(nn.predict(np.array([[0,0]], dtype=np.float))))
print('0 XOR 1:', np.round(nn.predict(np.array([[0,1]], dtype=np.float))))
print('1 XOR 0:', np.round(nn.predict(np.array([[1,0]], dtype=np.float))))
print('1 XOR 1:', np.round(nn.predict(np.array([[1,1]], dtype=np.float))))

# Batch-predict several inputs at once.
print(np.round(nn.predict(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float))))
