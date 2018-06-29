import sys
sys.path.append("../..")
import warnings
import time
import numpy as np

from lightenn import neuralnet
from lightenn import types

def scale_example(x, mu_sqft, mu_beds, std_sqft, std_beds):
    scaled_x = []
    scaled_x.append((x[0] - mu_sqft)/std_sqft)
    scaled_x.append((x[1] - mu_beds)/std_beds)
    return scaled_x

# Apply feature scaling to training set
def feature_scale(training_x):
    mu_sqft = training_x.mean(axis=0)[0]
    mu_beds = training_x.mean(axis=0)[1]
    std_sqft = training_x.std(axis=0)[0]
    std_beds = training_x.std(axis=0)[1]
    training_x_scaled = []
    for x in training_x:
        training_x_scaled.append(scale_example(x, mu_sqft, mu_beds, std_sqft, std_beds))
    training_x_scaled = np.array(training_x_scaled, dtype=np.float)
    return training_x_scaled, mu_sqft, mu_beds, std_sqft, std_beds

# Skip the header, and any blank lines
def load_training_data(path):
    
    training_x = []
    training_y = []
    
    header = False
    with open(path, 'r') as f:
        for line in f:
            if header == False:
                header = True
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            t = line.split(',')
            x = t[0:len(t)-1]
            y = t[len(t)-1:len(t)]
            training_x.append(x)
            training_y.append(y)

    assert (len(training_x) > 0), 'Error: empty training set.'
    
    return (np.array(training_x, dtype=np.float),
            np.array(training_y, dtype=np.float))

def split_test(training_x, training_y, test_perc):
    split_idx = int((1.0-test_perc)*len(training_x))
    return (training_x[0:split_idx], training_y[0:split_idx],
            training_x[split_idx:len(training_x)],
            training_y[split_idx:len(training_y)])

# Hyperparams
num_epochs = 1000
learning_rate = 0.2

# For selecting random weights and biases
mu = 0.0
stddev = 1.0

# Load and feature-scale the data
training_x, training_y = load_training_data('./data/Portland_House_Prices.csv')
training_x_scaled, mu_sqft, mu_beds, std_sqft, std_beds = feature_scale(training_x)

# Stabilize randomness
np.random.seed(123)

# Build neural net
nn = neuralnet.NeuralNet()
nn.add_input(2)
# Since this is a Linear Regression, we do not apply an activation function to the ouput node
nn.add_output(1, activation_type=types.ActivationType.NONE)
nn.initialize(loss_type=types.LossType.SQUARED_ERROR, learning_rate=learning_rate)

# Numerical gradient check
nn.check_gradients()

# Train in full-batch mode
t_0 = time.time()
nn.train_full((training_x_scaled, training_y), num_epochs)
t_1 = time.time()
tot_time = round(t_1 - t_0, 2)
print('Total time (in seconds):', tot_time)

# Compute closed form of parameters. Add column of ones:
training_x_bias = np.c_[np.ones(len(training_y)), training_x]
thetas = np.dot(np.linalg.inv(np.dot(training_x_bias.transpose(),
                                     training_x_bias)),
                np.dot(training_x_bias.transpose(), training_y))
thetas = thetas.reshape((len(thetas),))

# Test model predictions against closed form, to within an epsilon
epsilon = 1e-8
diff = False
for i, y in enumerate(training_y):
    x1_scaled = (training_x[i, 0] - mu_sqft)/std_sqft
    x2_scaled = (training_x[i, 1] - mu_beds)/std_beds
    y_hat = nn.predict(np.array([[x1_scaled, x2_scaled]], dtype=np.float))
    y_hat_closed = training_x[i, 0]*thetas[1] + training_x[i, 1]*thetas[2] + thetas[0]
    if abs(y_hat - y_hat_closed) > epsilon:
        warnings.warn('Predicted y_hat of', y_hat,
                      'differs from closed form of',
                      y_hat_closed, '.')
        diff = True
if diff == False:
    print('NN predictions match closed form to within', epsilon, '.')
