import os
import shutil
import json
import numpy as np
from lightenn import types

ALMOST_0 = 0.0000001
ALMOST_1 = 0.9999999

# TODO: this is not vectorized
def compute_example_loss(y, y_hat, layers, loss_type, regularizer):
    
    # Account for regularization, if present
    reg_term = 0.0
    if regularizer is not None:
        lambd = regularizer[1]
        if regularizer[0] == types.RegType.L1:
            for i in range(1, len(layers)):
                reg_term += lambd * np.sum(np.absolute(layers[i].wgts))
        elif regularizer[0] == types.RegType.L2:
            for i in range(1, len(layers)):
                reg_term += lambd * np.sum(np.power(layers[i].wgts, 2))
    
    # We return the sum to account for the multi-class case
    if loss_type == types.LossType.SQUARED_ERROR:
        return np.sum(np.power((y - y_hat), 2)) + reg_term
    elif loss_type == types.LossType.CROSS_ENTROPY:
        return np.sum(-1.0 * (y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))) + reg_term

# clamp the sigmoid
def sigmoid(x):
    
    v = 1.0 / (1.0 + np.exp(-x))
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

# For use with SGD
def sample_indices(index_list, size=1, replace=False):
    
    if size <= 0 or len(index_list) == 0:
        return [], index_list
    
    if size >= len(index_list):
        size = len(index_list)
    
    a = np.random.choice(len(index_list), size, replace=False)
    r = []
    for idx in a:
        r.append(index_list[idx])
    if replace == False:
        d = sorted(a, reverse=True)
        for idx in d:
            del index_list[idx]

    return r, index_list

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

