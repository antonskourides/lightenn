import numpy as np
from lightenn import types
from lightenn import layer

class NNBuilder:
    
    def __init__(self, layers, config):
        
        self.layers = layers
        self.config = {}
        self.layer_descriptors = []
    
    def add_input(self, size, values=None, dropout_p=0.0):
        
        assert (size > 0), 'Error: cannot add a layer of size 0.'
        assert (dropout_p >= 0.0 and dropout_p < 1.0), 'Error: dropout_p must be in the range [0.0, 1.0).'
        
        if values is not None:
            assert (isinstance(values, np.ndarray)), 'Error: values must be an ndarray.'
            assert (values.dtype == np.float), 'Error: values must be of type np.float.'
            assert (len(values.shape) == 1), 'Error: values ndarray must be 1-D.'
            assert (values.shape[0] == size), 'Error: size of values ndarray must match input layer size.'
        
        l_dict = {}
        l_dict['layer_type'] = types.LayerType.INPUT
        l_dict['size'] = size
        l_dict['dropout_p'] = dropout_p
        if values is not None:
            l_dict['values'] = values
        
        self.layer_descriptors.append(l_dict)

    def add_fully_connected(self, size, activation_type=types.ActivationType.SIGMOID,
                        wgts=None, biases=None, dropout_p=0.0):
        
        assert (size > 0), 'Error: cannot add a layer of size 0.'
        assert (dropout_p >= 0.0 and dropout_p < 1.0), 'Error: dropout_p must be in the range [0.0, 1.0).'
        self._check_wgts(wgts, size)
        self._check_biases(biases, size)
        self._add_layer(size, layer_type=types.LayerType.FULLY_CONNECTED,
                        activation_type=activation_type, wgts=wgts, biases=biases, dropout_p=dropout_p)
        
    def add_output(self, size, activation_type=types.ActivationType.SIGMOID, wgts=None, biases=None):
        
        assert (size > 0), 'Error: cannot add a layer of size 0.'
        self._check_wgts(wgts, size)
        self._check_biases(biases, size)
        self._add_layer(size, layer_type=types.LayerType.OUTPUT,
                        activation_type=activation_type, wgts=wgts, biases=biases)

    # Do your checks on neural net specification here
    def initialize(self, config):
    
        self.config = config
        loss_type = self.config['loss_type']
        regularizer = self.config['regularizer']
        
        assert (len(self.layer_descriptors) > 1), ('Error: neural network must have at least two layers.')
        assert (self.layer_descriptors[0]['layer_type'] == types.LayerType.INPUT), ('Error: first layer must be of type INPUT.')
        assert (self.layer_descriptors[len(self.layer_descriptors) - 1]['layer_type'] == types.LayerType.OUTPUT), ('Error: last layer must be of type OUTPUT.')
        assert isinstance(loss_type, types.LossType), ('Error: loss must be of type types.LossType')
        
        if regularizer is not None:
            assert (isinstance(regularizer, tuple) and (len(regularizer) == 2) and isinstance(regularizer[0], types.RegType) and isinstance(regularizer[1], float)), ('Error: regularizer must be a tuple of form (types.RegType, lambd).')

        hidden_layers = self.layer_descriptors[1:len(self.layer_descriptors)-1]
        for h in hidden_layers:
            assert (h['layer_type'] not in [types.LayerType.INPUT, types.LayerType.OUTPUT]), ('Error: hidden layers cannot be of type INPUT or OUTPUT.')

        # Loop through the layer descriptors and build the Layers.
        for ld_i, ld in enumerate(self.layer_descriptors):
            if ld_i == 0:
                self.layers.append(self._init_layer(ld, ld_i, None))
            else:
                prev_size = self.layer_descriptors[ld_i-1]['size']
                curr_size = self.layer_descriptors[ld_i]['size']
                if 'wgts' in ld:
                    wgts = ld['wgts']
                    assert (wgts.shape[0] == prev_size and wgts.shape[1] == curr_size), ('Error: weights must have shape (prev_size, curr_size)')
                self.layers.append(self._init_layer(ld, ld_i, prev_size))

        # Link all the layers together we can do forward() and backward() ops.
        for i in range(1, len(self.layers)):
            if i < len(self.layers) - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
            
    def _check_wgts(self, wgts, size):
        if wgts is not None:
            assert (isinstance(wgts, np.ndarray)), 'Error: wgts must be an ndarray.'
            assert (wgts.dtype == np.float), 'Error: wgts must be of type np.float.'
            assert (len(wgts.shape) == 2), 'Error: wgts ndarray must be 2-D.'
            assert (wgts.shape[1] == size), 'Error: wgts.shape[1] must match layer size.'

    def _check_biases(self, biases, size):
        if biases is not None:
            assert (isinstance(biases, np.ndarray)), 'Error: biases must be an ndarray.'
            assert (biases.dtype == np.float), 'Error: biases must be of type np.float.'
            assert (len(biases.shape) == 1), 'Error: biases ndarray must be 1-D.'
            assert (biases.shape[0] == size), 'Error: size of biases ndarray must match layer size.'

    def _add_layer(self, size, activation_type=types.ActivationType.SIGMOID,
                   layer_type=types.LayerType.FULLY_CONNECTED,
                   wgts=None, biases=None, dropout_p=0.0):
        
        l_dict = {}
        l_dict['size'] = size
        l_dict['activation_type'] = activation_type
        l_dict['layer_type'] = layer_type
        l_dict['dropout_p'] = dropout_p
        
        if wgts is not None:
            l_dict['wgts'] = wgts

        if biases is not None:
            l_dict['biases'] = biases
        
        self.layer_descriptors.append(l_dict)

    def _init_layer(self, lyr_desc, lyr_desc_i, prev_size):
    
        if lyr_desc['layer_type'] == types.LayerType.INPUT:
        
            if 'values' in lyr_desc:
                values = lyr_desc['values']
            else:
                values = np.zeros((lyr_desc['size'],), dtype=np.float)
            return layer.InputLayer(self.config, lyr_desc['size'], lyr_desc_i, values, dropout_p=lyr_desc['dropout_p'])
        
        else:
            
            curr_size = lyr_desc['size']
            activation_type = lyr_desc['activation_type']
            
            wgts = None
            if 'wgts' in lyr_desc:
                wgts = lyr_desc['wgts']
            
            biases = None
            if 'biases' in lyr_desc:
                biases = lyr_desc['biases']
            
            if lyr_desc['layer_type'] == types.LayerType.FULLY_CONNECTED:
                return layer.FullyConnectedLayer(self.config,
                                                 prev_size,
                                                 curr_size,
                                                 lyr_desc_i,
                                                 activation_type=activation_type,
                                                 wgts=wgts,
                                                 biases=biases,
                                                 dropout_p=lyr_desc['dropout_p'])
            else:
                return layer.OutputLayer(self.config,
                                         prev_size,
                                         curr_size,
                                         lyr_desc_i,
                                         activation_type=activation_type,
                                         wgts=wgts,
                                         biases=biases)


