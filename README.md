## LighteNN

Hello Machine Learning Community!

LighteNN is a lightweight Deep Learning library written from scratch, using
numpy and a handful of supporting packages. The backprop implementation was 
re-derived from first principles (except for the initial Cost functions.) No
copy/pasting of math or code!

Features supported so far:

- Cross-Entropy and Squared Error cost.
- Sigmoid, ReLU, and Linear activations.
- L1, L2, and Inverted Dropout regularization. Note: in this implementation, 
`dropout_p` is the probability of *dropping* a node, not keeping it.
- Stochastic Gradient Descent (SGD) training.
- Model persistence using the `serialize` package (just calling pickle 
underneath.)

Convolutional Neural Network (CNN) and GPU support coming soon!

Please see the ./usage_examples/ folder for sample use cases. Some toy examples
are included in:

1_toy_examples/

An MNIST training run is included in: 

2_mnist/

All the required training data is included with the examples. More examples to 
follow!

The code is (somewhat) vectorized but I'm sure true vectorization fanatics will
find a lot of room for improvement.

Thanks for stopping by!

Anton
