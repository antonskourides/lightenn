# LighteNN

LighteNN is a lightweight Neural Network library written from scratch, using `numpy` and a handful of standard modules. The backprop implementation was re-derived from scratch using first principles. 

### Requirements

LighteNN requires Python 3.6.x and `numpy` to run. The current implementation has been tested against `numpy 1.13.1`. All of the other modules required by LighteNN are contained in the Python 3.6 standard library.

### Features

Features supported so far:

- Cross-Entropy and Squared Error cost.
- Sigmoid, ReLU, and Linear activations.
- L1, L2, and Inverted Dropout regularization. Note: in this implementation, `dropout_p` is the probability of *dropping* a node, not keeping it.
- Full-Batch, Mini-Batch, and Stochastic Gradient Descent (SGD) training.
- Model persistence using the `serialize` package (just calling `pickle` underneath.)

Convolutional Neural Network (CNN) and GPU support coming soon!

### Usage Examples

Please see the ./usage_examples/ folder for sample use cases. 

Efficient MNIST training runs, using both a Multi-Layer Neural Net and a One-Layer Logistic Regression, are included in:

`mnist/`

A classical Linear Regression, using a One-Layer NN with a linear output, is performed in:

`linear/`

Feature selection using Logistic Regression Analysis is performed in:

`feature_selection`

Additional toy examples are included in:

`toy_examples/`

All the required training data is included with the examples. More examples to follow!

### Vectorization

The code is vectorized wherever feasible. The heart of the backprop implementation resides in the `lightenn.layers.vec` package, and a quick search there will show that no loops are present in the forward or backward passes.

This said, fully-vectorized code can be hard to read and understand, especially for people who are new to Machine Learning. To help with this, a non-vectorized implementation which produces identical output is also included in `lightenn.layers.novec`. 

*LighteNN always runs vectorized code by default*, however non-vectorized runs can be launched by passing `target=types.TargetType.NOVEC` to the `neuralnet.initialize()` method.

Please note that the non-vectorized code is included for explanatory purposes only, and is not appropriate for training runs involving all but the very smallest of datasets.

### The NeuralNet Class and Static Type-Checking

The `lightenn.neuralnet.NeuralNet` class exposes the main API for using LighteNN. Placeholders for static type-checking (valid from Python 3.6 forward) have been left in place in this class, in hopes that static type-checks will eventually be fully supported in Python :)

Thanks for stopping by!

Anton
