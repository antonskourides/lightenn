# LighteNN

Hello Machine Learning Community!

LighteNN is a lightweight Machine Learning library written from scratch, using
`numpy` and a handful of supporting packages. The backprop implementation was
re-derived from first principles (except for the initial Cost functions.) 

#### Features

Features supported so far:

- Cross-Entropy and Squared Error cost.
- Sigmoid, ReLU, and Linear activations.
- L1, L2, and Inverted Dropout regularization. Note: in this implementation, 
`dropout_p` is the probability of *dropping* a node, not keeping it.
- Full-Batch, Mini-Batch, and Stochastic Gradient Descent (SGD) training.
- Model persistence using the `serialize` package (just calling `pickle` 
underneath.)

Convolutional Neural Network (CNN) and GPU support coming soon!

#### Usage Examples

Please see the ./usage_examples/ folder for sample use cases. Some toy examples
are included in:

`1_toy_examples/`

A classical Linear Regression is performed in:

`2_linear/`

MNIST training runs, using both Logistic Regression and a Multi-Layer Neural Net,
are included in:

`3_mnist/`

All the required training data is included with the examples. More examples to 
follow!

#### Vectorization

The code is vectorized wherever feasible. The heart of the backprop implementation 
resides in the `lightenn.layers.vec` package, and a quick search there will show 
that *no loops are present* in the forward or backward passes.

This said, fully-vectorized code can be hard to read and understand, especially
for people who are new to Machine Learning. To help with this, a non-vectorized
implementation which produces identical output is also included in
`lightenn.layers.novec`. Non-vectorized runs can be launched by passing
`target=types.TargetType.NOVEC` to the `neuralnet.initialize()` method.

Please note that the non-vectorized code is included for explanatory purposes only,
and is not appropriate in training runs involving all but the very smallest of
data-sets.

#### The NeuralNet class and Static Type-Checking

The `lightenn.neuralnet.NeuralNet` class exposes the main API for using LighteNN.
Placeholders for static type-checking (valid from Python 3.6 forward) have been left 
in place in this class, in hopes that static type-checks will eventually be supported
fully in Python :)

Thanks for stopping by!

Anton
