# AutoGrad

[![Build Status](https://travis-ci.org/denizyuret/AutoGrad.jl.svg?branch=master)](https://travis-ci.org/denizyuret/AutoGrad.jl)

AutoGrad.jl is an automatic differentiation package for Julia.  It can
differentiate native Julia code that include loops, conditionals,
closures etc. and can handle higher order derivatives.

## Example

Here is a linear regression example simplified from
[housing.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/examples/housing.jl):

```
using AutoGrad

function loss(w)
    global xtrn,ytrn
    ypred = w[1]*xtrn .+ w[2]
    sum(abs2(ypred - ytrn)) / size(ypred,2)
end

function train(w; lr=.1, epochs=20)
    gradfun = grad(loss)
    for epoch=1:epochs
        g = gradfun(w)
        for i in 1:length(w)
            w[i] -= lr * g[i]
        end
    end
    return w
end
```

The `loss` function takes parameters as input and returns the loss to
be minimized.  The parameter `w` for this example is a pair: `w[1]` is
a weight matrix, and `w[2]` is a bias vector.  The training data
`xtrn,ytrn` are in global variables.  `ypred` is the predicted output,
and the last line computes the quadratic loss.  The `loss` function is
implemented in regular Julia.

The `train` function takes initial parameters and returns optimized
parameters.  `grad` is the only AutoGrad function used: it creates a
function `gradfun` that takes the same arguments as `loss`, but
returns the gradient instead.  The returned gradient will have the
same type and shape as the input argument.  The `for` loop implements
gradient descent, where we calculate the gradient and subtract a
scaled version of it from the weights.

See the [examples
directory](https://github.com/denizyuret/AutoGrad.jl/blob/master/examples)
for more examples, and the extensively documented
[core.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/core.jl)
for details.

## Code structure

[core.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/core.jl)
implements the main functionality with some support functions in
[util.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/collections.jl).
[collections.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/collections.jl)
adds support for Arrays, Tuples, and Dictionaries.  The numerical
gradients are defined in files such as `base/math.jl`,
`special/trig.jl` that mirror the organization under `julia/base`.

## Current status and future work

The gradient coverage is spotty, I am still adding more gradients to
cover the Julia base.  You can add your own primitives with gradients
as described in detail in
[core.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/core.jl).
Here is an example:

```
@primitive hypot
hypot(Grad{1}, y, x1, x2)=(dy->dy.*x1./y)
hypot(Grad{2}, y, x1, x2)=(dy->dy.*x2./y)
```

The `@primitive` macro marks `hypot` as a new primitive and the next
two lines define gradients wrt the first and second argument.

Next steps are to make models faster by providing support for
overwriting functions (memory allocation is slow) and GPU operations.
I should also find out about the efficiency of closures and untyped
functions in Julia which are used extensively in the code.

## Acknowledgments and references

AutoGrad.jl was written by [Deniz
Yuret](http://www.denizyuret.com). Large parts of the code are
directly ported from the Python
[autograd](https://github.com/HIPS/autograd) package.  I'd like to
thank autograd author Dougal Maclaurin for his support.  See [(Baydin
et al. 2015)](https://arxiv.org/pdf/1502.05767.pdf) for a general
review of automatic differentiation, [autograd
tutorial](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md)
for some Python examples, and Dougal's PhD thesis for design
principles.