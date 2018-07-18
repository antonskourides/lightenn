####################################################################
# Below we perform feature selection by logistic regression analysis.
#
# We run a logistic regression using a two-layer neural net. The
# input layer has size NUM_RANDOM + 2 (for the top and bottom pixels).
# The sigmoid output layer has 2 logits, corresponding to the classes
# "top_bright" and "bottom_bright".
#
# After training, we output a top-k of the features with the highest
# attached weights. We expect our top and bottom pixel features to be
# the top-2 features.

import sys
sys.path.append("../..")
import numpy as np
import random
import time

from lightenn import neuralnet
from lightenn import types
from lightenn import utils

NUM_EXAMPLES = 10000    # number of training examples to generate
MIN = 0.                # min pixel value
MAX = 255.              # max pixel value
LOW_THRESH = 20.        # a pixel with a value less than 20 is not bright
HIGH_THRESH = 235.      # a pixel with a value exceeding 235 is bright
NUM_RANDOM = 100        # the number of random pixels between the top and bottom pixel

# Generate a training example with a bright bottom pixel,
# a non-bright top pixel, and NUM_RANDOM randomly-valued
# pixels in between
def generate_bot_bright():
    top_pixel = np.random.randint(int(MIN), int(LOW_THRESH) + 1, size=(1,))
    bot_pixel = np.random.randint(int(HIGH_THRESH), int(MAX) + 1, size=(1,))
    rand_pixels = np.random.randint(int(MIN), int(MAX) + 1, size=(NUM_RANDOM,))
    pixels = np.concatenate((top_pixel, rand_pixels, bot_pixel))
    pixels = pixels.astype(str)
    # Return a complete training example, including the two y-values
    return ','.join(pixels.tolist()) + ',0,1'

# Generate a training example with a bright top pixel,
# a non-bright bottom pixel, and NUM_RANDOM randomly-valued
# pixels in between
def generate_top_bright():
    top_pixel = np.random.randint(int(HIGH_THRESH), int(MAX) + 1, size=(1,))
    bot_pixel = np.random.randint(int(MIN), int(LOW_THRESH) + 1, size=(1,))
    rand_pixels = np.random.randint(int(MIN), int(MAX) + 1, size=(NUM_RANDOM,))
    pixels = np.concatenate((top_pixel, rand_pixels, bot_pixel))
    pixels = pixels.astype(str)
    # Return a complete training example, including the two y-values
    return ','.join(pixels.tolist()) + ',1,0'

def generate_header(num_random=NUM_RANDOM):
    hlist = []
    for i in range(num_random + 2):
        hlist.append('PIXEL_' + str(i))
    hlist = hlist + ['Y_0', 'Y_1']
    return ','.join(hlist) + "\n"

def generate_data(path, num_examples=10000):
    with open(path, 'w') as f:
        f.write(generate_header())
        for i in range(num_examples):
            type = random.randint(0,1)
            s = None
            if type == 0:
                s = generate_bot_bright()
            elif type == 1:
                s = generate_top_bright()
            f.write(s + "\n")

def split_validation(training_x, training_y, val_perc):
    split_idx = int((1.0-val_perc)*len(training_x))
    return (training_x[0:split_idx], training_y[0:split_idx],
            training_x[split_idx:len(training_x)], training_y[split_idx:len(training_y)])

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
            training_x.append(t[0:len(t)-2])
            training_y.append(t[len(t)-2:len(t)])

    assert (len(training_x) > 0), 'Error: empty training set.'

    return (np.array(training_x, dtype=np.float),
            np.array(training_y, dtype=np.float))

# Return the indices of the top-k features with the highest weights
def top_k_features(wgts, k=3):
    s = np.sum(np.absolute(wgts), axis=1)
    return np.argsort(-s)[:k]

# Stabilize randomness
np.random.seed(123)

# Generate traning data
generate_data('./data.csv')

# Percentage of validation data
val_perc = 0.2

# Load training and validation sets
training_x, training_y = load_training_data('./data.csv')
training_x, training_y, validation_x, validation_y = split_validation(training_x, training_y, val_perc)

# Rescale your training data
training_x = utils.rescale(training_x, MIN, MAX, 0., 1.)
validation_x = utils.rescale(validation_x, MIN, MAX, 0., 1.)

# Hyperparams
num_epochs = 100
learning_rate = 1.0 # high learning rate!

# Build neural net
nn = neuralnet.NeuralNet()
nn.add_input(NUM_RANDOM + 2)
nn.add_output(2, activation_type=types.ActivationType.SIGMOID)
nn.initialize(loss_type=types.LossType.CROSS_ENTROPY, learning_rate=learning_rate)

# Train in full-batch mode
t_0 = time.time()
nn.train_full((training_x, training_y), num_epochs,
              validation_set=(validation_x, validation_y))
t_1 = time.time()
tot_time = round(t_1 - t_0, 2)
print('Total time (in seconds):', tot_time)

# Look at the weights after training. All of the weights attached
# to useless features/pixels will have low pos+ or neg- values.
print(nn.layers[1].wgts)

# Look at the top-k features in terms of their attached weights.
# We expect the indices of our top and bottom pixels to be the
# top two:
print('Top-k features:')
print(top_k_features(nn.layers[1].wgts, k=5))
