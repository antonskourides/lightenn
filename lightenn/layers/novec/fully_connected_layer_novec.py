import numpy as np
from lightenn.layers.vec import fully_connected_layer

class FullyConnectedLayerNoVec(fully_connected_layer.FullyConnectedLayer):

    def compute_cost_wrt_activations(self, batch_size, y, y_hat):

        for x in range(batch_size):
            for act_idx in range(self.size):
                for next_act_idx in range(self.next.size):
                    # obtain the grad of the cost wrt act_i+1
                    next_grad = self.next.cost_wrt_activations[x, next_act_idx]
                    # obtain the grad of the act_i+1 wrt act_i
                    next_wgt = self.next.wgts[
                        act_idx, next_act_idx]  # next_wgt is the weight that connects my act_i to act_i+1
                    next_z = self.next.z_vals[x, next_act_idx]  # next_z is the z-value for the act_i+1
                    curr_grad = self.activation_prime(next_z, self.next.activation_type) * next_wgt
                    # if there are multiple act_i+1 that are connected to act_i, we must add
                    self.cost_wrt_activations[x, act_idx] += next_grad * curr_grad

        # Dropout: gradient of cost w.r.t. my activations must
        # also now include the dropout mask and inverse terms that are
        # associated with my layer
        return np.divide(np.multiply(self.cost_wrt_activations, self.dropout_mask), 1.0 - self.dropout_p)

    def backward_compute_grads(self, y, y_hat):

        # Batch size that was fed in at the input layer. Because the dot product
        # is used for every forward, this will always be the value of the first
        # dimension of the activation layer
        batch_size = self.activations.shape[0]

        # Partial derivatives of cost w.r.t. my activations
        self.cost_wrt_activations = np.zeros((batch_size, self.size), dtype=np.float)

        # Partial derivatives of cost w.r.t. weights
        self.cost_wrt_wgts = np.zeros((self.prev.size, self.size), dtype=np.float)

        # Partial derivatives of cost w.r.t. biases
        self.cost_wrt_biases = np.zeros((self.size), dtype=np.float)

        # Partial derivatives of my activations w.r.t. my incoming wgts
        acts_wrt_wgts = np.zeros((batch_size, self.prev.size, self.size), dtype=np.float)

        # Partial derivatives of my activations w.r.t. my incoming biases
        acts_wrt_bias = np.zeros((batch_size, self.size), dtype=np.float)

        # Compute gradient of my activations wrt wgts and biases
        for x in range(batch_size):
            z_vals = self.z_vals[x]
            z_vals_prime = self.activation_prime(z_vals, self.activation_type)
            for act_idx in range(self.size):
                z_val_prime = z_vals_prime[act_idx]
                for prev_idx in range(self.prev.size):
                    in_act = self.prev.activations[x, prev_idx]
                    acts_wrt_wgts[x, prev_idx, act_idx] = z_val_prime * in_act
                acts_wrt_bias[x, act_idx] = z_val_prime

        # compute grad of cost wrt my activations
        self.compute_cost_wrt_activations(batch_size, y, y_hat)

        # compute grad of cost wrt wgts and bias
        for i in range(len(self.wgts)):
            for j in range(len(self.wgts[0])):
                self.cost_wrt_wgts[i, j] = 0.
                for x in range(batch_size):
                    self.cost_wrt_wgts[i, j] += self.cost_wrt_activations[x, j] * acts_wrt_wgts[x, i, j]

        for j in range(self.size):
            self.cost_wrt_biases[j] = 0.
            for x in range(batch_size):
                self.cost_wrt_biases[j] += self.cost_wrt_activations[x, j] * acts_wrt_bias[x, j]

        # Add partial regularization term, as applicable
        partial_reg_term = self.compute_partial_reg_term()
        self.cost_wrt_wgts = np.add(self.cost_wrt_wgts, partial_reg_term)

    def backward_adjust(self):

        learning_rate = self.nn_config['learning_rate']

        for i in range(len(self.wgts)):
            for j in range(len(self.wgts[0])):
                self.wgts[i, j] += -1. * self.cost_wrt_wgts[i, j] * learning_rate

        for i in range(len(self.biases)):
            self.biases[i] += -1. * self.cost_wrt_biases[i] * learning_rate
