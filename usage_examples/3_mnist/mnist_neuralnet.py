import sys
sys.path.append("../..")
import time
import pickle
import os
import numpy as np

from lightenn import neuralnet
from lightenn import types
from lightenn import utils
from lightenn import serialize

def load_training_data(x_pkl_path, y_pkl_path):

    training_x = None
    training_y = None

    with open(x_pkl_path, 'rb') as f:
        training_x = pickle.load(f)
    with open(y_pkl_path, 'rb') as f:
        training_y = pickle.load(f)
    
    # Flatten your 28 * 28 images to 784, and z-scale normalize
    training_x = training_x.reshape(training_x.shape[0], training_x.shape[1]*training_x.shape[2])
    training_x = utils.z_scale_normalize(training_x.astype(np.float))
    training_y = training_y.astype(np.float)
    return training_x, training_y

def split_validation(training_x, training_y, val_perc):
    split_idx = int((1.0-val_perc)*len(training_x))
    return (training_x[0:split_idx], training_y[0:split_idx],
            training_x[split_idx:len(training_x)], training_y[split_idx:len(training_y)])

# Hyperparams
num_epochs = 10
#learning_rate = 0.001
learning_rate = 0.01

# For selecting random weights and biases
mu = 0.0
stddev = 1.0

# Percentage of validation data
val_perc = 0.2

# Delta and Epsilon for gradient checking
delta = 0.0001
epsilon = 1e-8

mnist_data_root = './data/'
training_x, training_y = load_training_data(mnist_data_root + 'mnist_train_x.pkl',
                                            mnist_data_root + 'mnist_train_y.pkl')
training_x, training_y, validation_x, validation_y = split_validation(training_x, training_y, val_perc)
test_x, test_y = load_training_data(mnist_data_root + 'mnist_test_x.pkl', mnist_data_root + 'mnist_test_y.pkl')

model_path = './mnist_model.pkl'
np.random.seed(1234)
np.seterr(all='ignore')
nn = None

# If a model is already on disk, restore it. Otherwise, train
if os.path.exists(model_path) and os.path.isfile(model_path):
    nn = serialize.restore(model_path)
else:
    # Build neural net
    print('Building ...')
    nn = neuralnet.NeuralNet()
    nn.add_input(784)
    nn.add_fully_connected(64, activation_type=types.ActivationType.RELU)
    nn.add_output(10, activation_type=types.ActivationType.SIGMOID)
    nn.initialize(loss_type=types.LossType.CROSS_ENTROPY, learning_rate=learning_rate)
    # Train
    print('Training ...')
    t_0 = time.time()
    nn.train_sgd((training_x, training_y), num_epochs, validation_set=(validation_x, validation_y))
    t_1 = time.time()
    tot_time = round(t_1 - t_0, 2)
    print('Total time (in seconds):', tot_time)
    serialize.save(nn, model_path)

# Output test accuracy
print('Testing ...')
print('Test accuracy:', nn.get_accuracy((test_x, test_y)))

# Predict some examples
rand_examples = np.random.choice(list(range(0, len(test_x))), size=20, replace=False)
for i in rand_examples:
    p = np.argmax(nn.predict(np.array([test_x[i]])))
    a = np.argmax(np.array([test_y[i]]))
    print('Predicted:', p, 'Actual:', a)
