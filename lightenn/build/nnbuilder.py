from lightenn import types
from lightenn.layers.vec import output_layer, fully_connected_layer, input_layer
from lightenn.layers.novec import fully_connected_layer_novec, output_layer_novec

class NNBuilder:

    def __init__(self, layers, config):

        self.layers = layers
        self.config = config
        self.layer_descriptors = []

    def add_input(self, size, values=None, dropout_p=0.0):

        l_dict = {}
        l_dict['layer_type'] = types.LayerType.INPUT
        l_dict['size'] = size
        l_dict['dropout_p'] = dropout_p
        if values is not None:
            l_dict['values'] = values
        else:
            l_dict['values'] = None

        self.layer_descriptors.append(l_dict)

    def add_fully_connected(self, size, activation_type=types.ActivationType.SIGMOID,
                            wgts=None, biases=None, dropout_p=0.0):

        l_dict = {}
        l_dict['layer_type'] = types.LayerType.FULLY_CONNECTED
        l_dict['size'] = size
        l_dict['activation_type'] = activation_type
        l_dict['dropout_p'] = dropout_p

        if wgts is not None:
            l_dict['wgts'] = wgts
        else:
            l_dict['wgts'] = None

        if biases is not None:
            l_dict['biases'] = biases
        else:
            l_dict['biases'] = None

        self.layer_descriptors.append(l_dict)

    def add_output(self, size, activation_type=types.ActivationType.SIGMOID,
                   wgts=None, biases=None):

        l_dict = {}
        l_dict['layer_type'] = types.LayerType.OUTPUT
        l_dict['size'] = size
        l_dict['activation_type'] = activation_type

        if wgts is not None:
            l_dict['wgts'] = wgts
        else:
            l_dict['wgts'] = None

        if biases is not None:
            l_dict['biases'] = biases
        else:
            l_dict['biases'] = None

        self.layer_descriptors.append(l_dict)

    # Initialize the neural net:
    # 1) Call the constructor for each layer.
    # 2) Do basic checks on neural net specification.
    # 3) Link all layers together, using prev and next.
    # 4) Call each layer's initialize(). Each of these will do individual layer-level checks as appropriate.
    def initialize(self):

        for ld_i, ld in enumerate(self.layer_descriptors):
            if ld['layer_type'] == types.LayerType.INPUT:
                self.layers.append(
                    input_layer.InputLayer(self.config, ld['size'], ld_i, values=ld['values'],
                                           dropout_p=ld['dropout_p']))
            elif ld['layer_type'] == types.LayerType.FULLY_CONNECTED:
                if self.config['target'] == types.TargetType.NOVEC:
                    self.layers.append(
                        fully_connected_layer_novec.FullyConnectedLayerNoVec(self.config, ld['size'], ld_i,
                                                                             activation_type=ld[
                                                                                 'activation_type'],
                                                                             wgts=ld['wgts'],
                                                                             biases=ld['biases'],
                                                                             dropout_p=ld['dropout_p']))
                else:
                    self.layers.append(fully_connected_layer.FullyConnectedLayer(self.config, ld['size'], ld_i,
                                                                                 activation_type=ld[
                                                                                     'activation_type'],
                                                                                 wgts=ld['wgts'],
                                                                                 biases=ld['biases'],
                                                                                 dropout_p=ld['dropout_p']))
            elif ld['layer_type'] == types.LayerType.OUTPUT:
                if self.config['target'] == types.TargetType.NOVEC:
                    self.layers.append(output_layer_novec.OutputLayerNoVec(self.config, ld['size'], ld_i,
                                                                           activation_type=ld['activation_type'],
                                                                           wgts=ld['wgts'], biases=ld['biases']))
                else:
                    self.layers.append(output_layer.OutputLayer(self.config, ld['size'], ld_i,
                                                                activation_type=ld['activation_type'],
                                                                wgts=ld['wgts'], biases=ld['biases']))

        assert (len(self.layer_descriptors) > 1), ('Error: neural network must have at least two layers.')
        assert (isinstance(self.layers[0], input_layer.InputLayer)), (
            'Error: first layer must be of type INPUT.')
        assert (isinstance(self.layers[len(self.layers) - 1], output_layer.OutputLayer) or
                isinstance(self.layers[len(self.layers) - 1], output_layer_novec.OutputLayerNoVec)), (
            'Error: last layer must be of type OUTPUT.')

        # Ensure that all of the hidden layers are of type FULLY_CONNECTED
        for i in range(1, len(self.layers) - 1):
            assert (isinstance(self.layers[i],
                               fully_connected_layer.FullyConnectedLayer)), 'Error: hidden layers must be of type FULLY_CONNECTED.'

        # Connect all the layers together using prev and next
        self.layers[0].next = self.layers[1]
        for i in range(1, len(self.layers)):
            if i < len(self.layers) - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            else:
                self.layers[i].prev = self.layers[i - 1]

        # Run initialize() on each individual layer
        for i in range(len(self.layers)):
            self.layers[i].initialize()
