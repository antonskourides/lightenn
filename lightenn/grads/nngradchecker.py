import warnings
import numpy as np
from lightenn import utils
from lightenn import types

class NNGradChecker:
    
    def __init__(self, layers, config):
        self.layers = layers
        self.config = config

    # Check the gradient of loss w.r.t. a given weight or the bias (numerically).
    # There is no need to reset the whole network for every gradient check - as
    # long as the network input is the same, the output will be the same,
    # since we are not changing the weights in the backward pass. We just need to
    # be sure to set any weights that we perturb back to their original values.
    #
    # TODO: A full numerical gradient check on a large network can take a very
    # long time. Add a 'stochastic' parameter.
    def check_gradients(self, delta, epsilon, y=None, verbose=False, quit_on_fail=True):
    
        for l_i in range(1, len(self.layers)):
            l = self.layers[l_i]
            for i in range(l.wgts.shape[0]):
                for j in range(l.wgts.shape[1]):
                    if (not self.gradient_check(l_i, types.GradType.WEIGHT,
                                               (i,j), delta, epsilon, y=y,
                                               verbose=verbose) and
                        quit_on_fail == True):
                        exit(1)
            for i in range(l.biases.shape[0]):
                if (not self.gradient_check(l_i, types.GradType.BIAS, (i,), delta,
                                        epsilon, y=y, verbose=verbose) and
                    quit_on_fail == True):
                    exit(1)

    def gradient_check(self, layer_idx, type, idx_tuple, delta, epsilon, y=None, verbose=False):
        
        # Number of examples currently in input values
        y_rows = self.layers[0].values.shape[0]
        
        # Size of output layer
        y_cols = self.layers[len(self.layers)-1].size
        
        if y is None:
            y = np.ones((y_rows, y_cols), dtype=np.float)
        else:
            assert (isinstance(y, np.ndarray) and len(y.shape) == 2), 'Error: y must be a 2-D ndarray.'
            assert (y.shape == (y_rows, y_cols)), 'Error: y must have shape (num_examples, output_lyr_size).'
            
        assert (layer_idx in range(1,len(self.layers))), ('Error: layer_idx must be hidden or output layer for gradient check.')
        assert (isinstance(type, types.GradType)), ('Error: gradient check type must be of type types.GradType.')
        
        if type == types.GradType.WEIGHT:
            assert (len(idx_tuple) == 2), ('Error: weight gradient check requires tuple of size 2.')
            assert (idx_tuple[0] <= self.layers[layer_idx].wgts.shape[0] and idx_tuple[1] <= self.layers[layer_idx].wgts.shape[1]), 'Error: weight index out of range in gradient check.'
            return self._gradient_check_wgt(layer_idx, idx_tuple[0], idx_tuple[1], y, delta, epsilon, verbose=verbose)
        else:
            assert (len(idx_tuple) == 1), ('Error: bias gradient check requires tuple of size 1.')
            assert (idx_tuple[0] <= self.layers[layer_idx].biases.shape[0]), 'Error: bias index out of range in gradient check.'
            return self._gradient_check_bias(layer_idx, idx_tuple[0], y, delta, epsilon, verbose=verbose)

    # Compute an approximate linear gradient with rise over run using
    # two very nearby points
    def _grad_approx(self, x1, y1, x2, y2):
        return (y2-y1)/(x2-x1)

    def _gradient_check_wgt(self, layer_idx, i, j, y, delta, epsilon, verbose=False):
    
        layer = self.layers[layer_idx]
        w_orig = layer.wgts[i,j]
        w_pos = w_orig + delta
        w_neg = w_orig - delta
        
        # First, obtain backprop gradient
        for l in self.layers:
            l.forward()
        y_hat = self.layers[len(self.layers)-1].activations
        for k in range(len(self.layers)-1, 0, -1):
            self.layers[k].backward_compute_grads(y, y_hat)
        grad_backprop = layer.cost_wrt_wgts[i,j]
        
        # Perturb weight upwards and compute y-hat
        layer.wgts[i,j] = w_pos
        for l in self.layers:
            l.forward()
        y_hat_pos = self.layers[len(self.layers)-1].activations
        loss_pos = utils.compute_batched_loss(y, y_hat_pos, self.layers, self.config['loss_type'], self.config['regularizer'])
        layer.wgts[i,j] = w_orig
        
        # Perturb weight downwards and compute y-hat
        layer.wgts[i,j] = w_neg
        for l in self.layers:
            l.forward()
        y_hat_neg = self.layers[len(self.layers)-1].activations
        loss_neg = utils.compute_batched_loss(y, y_hat_neg, self.layers, self.config['loss_type'], self.config['regularizer'])
        layer.wgts[i,j] = w_orig
        
        # Compute numerical gradient between perturbed weights
        # using rise over run
        grad_numerical = self._grad_approx(w_neg, loss_neg,
                                           w_pos, loss_pos)

        if verbose == True:
            print('layer', layer_idx, 'weight:', i, j)
            print("w_orig:\t", w_orig)
            print("w_pos:\t", w_pos)
            print("w_neg:\t", w_neg)
            print("loss_pos:\t", loss_pos)
            print("loss_neg:\t", loss_neg)
            print("grad_numerical:\t", grad_numerical)
            print("grad_backprop:\t", grad_backprop)

        # Output warning if backprop grad differs from numerical grad in excess of some threshold
        diff = grad_backprop - grad_numerical
        if abs(diff) > epsilon:
            warn_str = ("Weight [%d, %d] in layer %d: diff of %.4E"
                        " is larger than epsilon of %.4E."
                        " Numerical grad: %.4E."
                        " Backprop grad: %.4E."
                        %(i, j, layer_idx, abs(diff), epsilon, grad_numerical, grad_backprop))
            warnings.warn(warn_str)
            return False
                                                                
        return True

    def _gradient_check_bias(self, layer_idx, i, y, delta, epsilon, verbose=False):
    
        layer = self.layers[layer_idx]
        b_orig = layer.biases[i]
        b_pos = b_orig + delta
        b_neg = b_orig - delta
        
        # First, obtain backprop gradient
        for l in self.layers:
            l.forward()
        y_hat = self.layers[len(self.layers)-1].activations
        for k in range(len(self.layers)-1, 0, -1):
            self.layers[k].backward_compute_grads(y, y_hat)
        grad_backprop = layer.cost_wrt_biases[i]
        
        # Perturb bias upwards and compute y-hat
        layer.biases[i] = b_pos
        for l in self.layers:
            l.forward()
        y_hat_pos = self.layers[len(self.layers)-1].activations
        loss_pos = utils.compute_batched_loss(y, y_hat_pos, self.layers, self.config['loss_type'], self.config['regularizer'])
        layer.biases[i] = b_orig
        
        # Perturb bias downwards and compute y-hat
        layer.biases[i] = b_neg
        for l in self.layers:
            l.forward()
        y_hat_neg = self.layers[len(self.layers)-1].activations
        loss_neg = utils.compute_batched_loss(y, y_hat_neg, self.layers, self.config['loss_type'], self.config['regularizer'])
        layer.biases[i] = b_orig
        
        # Compute numerical gradient between perturbed biases
        # using rise over run
        grad_numerical = self._grad_approx(b_neg, loss_neg,
                                           b_pos, loss_pos)

        if verbose == True:
            print('layer', layer_idx, 'bias:', i)
            print("b_orig:\t", b_orig)
            print("b_pos:\t", b_pos)
            print("b_neg:\t", b_neg)
            print("loss_pos:\t", loss_pos)
            print("loss_neg:\t", loss_neg)
            print("grad_numerical:\t", grad_numerical)
            print("grad_backprop:\t", grad_backprop)

        # Output warning if backprop grad differs from numerical grad in excess of some threshold
        diff = grad_backprop - grad_numerical
        if abs(diff) > epsilon:
            warn_str = ("Bias %d in layer %d: diff of %.4E"
                        " is larger than epsilon of %.4E."
                        " Numerical grad: %.4E."
                        " Backprop grad: %.4E."
                        %(i, layer_idx, abs(diff), epsilon, grad_numerical, grad_backprop))
            warnings.warn(warn_str)
            return False
                                                                
        return True




