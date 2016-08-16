# Examples

## [Housing](https://github.com/denizyuret/AutoGrad.jl/blob/master/examples/housing.jl)

This example uses the Housing dataset from the UCI Machine Learning
Repository to demonstrate a linear regression model. The dataset has
housing related information for 506 neighborhoods in Boston from
1978. Each neighborhood has 14 attributes, the goal is to use the
first 13, such as average number of rooms per house, or distance to
employment centers, to predict the 14’th attribute: median dollar
value of the houses.

To run the demo, simply `include("housing.jl")` and run `Housing.train()`.
The dataset will be automatically downloaded.  You can provide the
initial weights as an optional argument, which should be a pair of
1x13 weight matrix and a scalar bias.  `train` also accepts the
following keyword arguments: `lr` specifies the learning rate,
`epochs` gives number of epochs, and `seed` specifies the random
number seed.  The quadratic loss for the train and test sets will be
printed at every epoch and optimized parameters will be returned.

## [MNIST](https://github.com/denizyuret/AutoGrad.jl/blob/master/examples/mnist.jl)

This example learns to classify hand-written digits from the MNIST
dataset.  There are 60000 training and 10000 test examples. Each input
x consists of 784 pixels representing a 28x28 image.  The pixel values
have been normalized to [0,1]. Each output y is a ten-dimensional
one-hot vector (a vector that has a single non-zero component)
indicating the correct class (0-9) for a given image.

To run the demo, simply `include("mnist.jl")` and run `MNIST.train()`.
The dataset will be automatically downloaded.  You can provide the
initial weights as an optional argument to `train`, which should have
the form [w0,b0,w1,b1,...] where wi (with size = output x input) is
the weight matrix and bi (with size = output) is the bias vector for
layer i.  The function `MNIST.weights(h...)` can be used to create
random starting weights for a neural network with hidden sizes (h...).
If not specified, default weights are created using `MNIST.weights()`
which correspond to a 0 hidden layer network, i.e. a softmax model.
`train` also accepts the following keyword arguments: `lr` specifies
the learning rate, `epochs` gives number of epochs.  The cross entropy
loss and accuracy for the train and test sets will be printed at every
epoch and optimized parameters will be returned.
