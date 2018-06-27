from lightenn import types
from lightenn.layers.vec import fully_connected_layer


class OutputLayer(fully_connected_layer.FullyConnectedLayer):

    def __init__(self, nn_config, size, idx,
                 activation_type=types.ActivationType.SIGMOID,
                 wgts=None, biases=None):
        super().__init__(nn_config, size, idx,
                         activation_type=activation_type,
                         wgts=wgts, biases=biases)

    def compute_loss_term(self, m, y, y_hat):
        if self.nn_config['loss_type'] == types.LossType.CROSS_ENTROPY:
            return (-1.0 / m) * (y / y_hat - (1.0 - y) / (1.0 - y_hat))
        elif self.nn_config['loss_type'] == types.LossType.SQUARED_ERROR:
            return (-1. / m) * (y - y_hat)

    def compute_cost_wrt_activations(self, batch_size, y, y_hat):
        return self.compute_loss_term(batch_size, y, y_hat)
