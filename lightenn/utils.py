import os
import shutil
import json
import numpy as np

from lightenn import types

ALMOST_0 = 0.0000001
ALMOST_1 = 0.9999999

# clamp the sigmoid
def sigmoid(x):
    
    v = 1.0 / (1.0 + np.exp(-x))
    
    # Remove this when vectorized
    if isinstance(x, np.float):
        if v < ALMOST_0:
            v = ALMOST_0
        elif v > ALMOST_1:
            v = ALMOST_1
        return v

    v[v < ALMOST_0] = ALMOST_0
    v[v > ALMOST_1] = ALMOST_1
    
    return v

def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

def relu(x):
    return np.maximum(x, 0.0)

# From here:
# https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
def relu_prime(x):
    return (x > 0.0) * 1.0

def z_scale_normalize(data):
    mu = np.mean(data)
    stddev = np.std(data)
    return np.divide(np.subtract(data, mu), stddev)

def rescale(data, old_min, old_max, new_min, new_max):
    return new_min + (data - old_min)*(new_max - new_min)/(old_max - old_min)

def mkdir(dirpath, clear=True):
    if clear == True:
        rmdir(dirpath)
    os.makedirs(dirpath)

def rmdir(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4, sort_keys=True)

def compute_batched_loss(batch_y, batch_y_hat, layers, loss_type, regularizer):

    batch_size = batch_y.shape[0]
    loss = 0.

    # Account for regularization, if present
    reg_term = 0.0
    if regularizer is not None:
        lambd = regularizer[1]
        if regularizer[0] == types.RegType.L1:
            for i in range(1, len(layers)):
                reg_term += (lambd/batch_size) * np.sum(np.absolute(layers[i].wgts))
        elif regularizer[0] == types.RegType.L2:
            for i in range(1, len(layers)):
                reg_term += (lambd/(2.0*batch_size)) * np.sum(np.power(layers[i].wgts, 2))

    # We return the sum to account for the multi-class case
    if loss_type == types.LossType.SQUARED_ERROR:
        loss = np.sum(np.power((batch_y - batch_y_hat), 2))
        return (1. / (2. * batch_size)) * loss + reg_term
    elif loss_type == types.LossType.CROSS_ENTROPY:
        loss = np.sum(-1.0 * (batch_y * np.log(batch_y_hat) + (1.0 - batch_y) * np.log(1.0 - batch_y_hat)))
        return (1. / batch_size) * loss + reg_term
