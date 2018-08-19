# AutoGrad

<!--
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_0.6.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_0.7.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_1.0.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
-->

[![Build Status](https://travis-ci.org/denizyuret/AutoGrad.jl.svg?branch=master)](https://travis-ci.org/denizyuret/AutoGrad.jl)
[![coveralls](https://coveralls.io/repos/github/denizyuret/AutoGrad.jl/badge.svg?branch=master)](https://coveralls.io/github/denizyuret/AutoGrad.jl?branch=master)
[![codecov](https://codecov.io/gh/denizyuret/AutoGrad.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/denizyuret/AutoGrad.jl)

AutoGrad.jl is an automatic differentiation package for Julia.  It is
based on the popular Python
[autograd](https://github.com/HIPS/autograd) package and forms the
foundation of the [Knet](https://github.com/denizyuret/Knet.jl) Julia
deep learning framework.  AutoGrad can differentiate regular Julia
code that includes loops, conditionals, helper functions, closures
etc. by keeping track of the primitive operations and using this
execution trace to compute gradients.  It uses reverse mode
differentiation (a.k.a. backpropagation) so it can efficiently handle
functions with array inputs and scalar outputs.  It can compute
gradients of gradients to handle higher order derivatives.  Please see
the comments in
[core.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/core.jl)
for a description of how the code works in detail.

## Installation

You can install AutoGrad in Julia using:
```
julia> using Pkg; Pkg.add("AutoGrad")
```

In order to use it in your code start with:
```
using AutoGrad
```

## Example

Here is a linear regression example simplified from
[housing.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/examples/housing.jl):

```
using AutoGrad

function loss(w)
    global xtrn,ytrn
    ypred = w[1]*xtrn .+ w[2]
    sum(abs2, ypred - ytrn) / size(ypred,2)
end

function train(w; lr=.1, epochs=20)
    lossgrad = grad(loss)
    for epoch=1:epochs
        g = lossgrad(w)
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
function `lossgrad` that takes the same arguments as `loss`, but
returns the gradient instead.  The returned gradient will have the
same type and shape as the input argument.  The `for` loop implements
gradient descent, where we calculate the gradient and subtract a
scaled version of it from the weights.

See the [examples
directory](https://github.com/denizyuret/AutoGrad.jl/blob/master/examples)
for more examples, and the extensively documented
[core.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/core.jl)
for details.

## Extending AutoGrad

AutoGrad can only handle a function if the primitives it uses have
known gradients.  You can add your own primitives with gradients as
described in detail in
[core.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/core.jl)
or using the `@primitive` and `@zerograd` macros in
[macros.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/macros.jl)
Here is an example:

```
@primitive hypot(x1,x2),dy,y  (dy.*x1./y)  (dy.*x2./y)
```

The `@primitive` macro marks the `hypot(::Any,::Any)` method as
a new primitive and the next two expressions define gradient functions
wrt the first and second argument.  The gradient expressions can refer
to the parameters `(x1,x2)`, the return variable `y` and its gradient
`dy` (optionally indicated after the argument list) in the method
declaration.

Note that Julia supports multiple-dispatch, i.e. a function may have
multiple methods each supporting different argument types.  For
example `hypot(x1::Number,x2::Number)` and
`hypot(x1::Array,x2::Array)` are two different hypot methods.  In
AutoGrad.jl each method can independently be defined as a primitive
and can have its own specific gradient. Generally AutoGrad defines
gradients without using argument types to keep the rules generic.

## Code structure

[core.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/core.jl)
implements the main functionality and acts as the main documentation
source.
[macros.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/macros.jl)
has some support functions to define and test new primitives.
[getindex.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/getindex.jl),
[iterate.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/iterate.jl) and
[cat.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/cat.jl)
set up support for common data structures including Arrays, Tuples,
and Dictionaries.  The numerical gradients are defined in files such
as `base.jl` and `math.jl`.

## Current status and future work

The gradient coverage and unit testing are spotty, I am still adding
more gradients and tests to cover the Julia base. Documentation needs
to be improved. Overwriting functions (e.g. `setindex!`) are not
supported. Efficiency could be improved by reducing runtime
compilation, memoization, and support for static computation.

## Acknowledgments and references

AutoGrad.jl was written by [Deniz
Yuret](http://www.denizyuret.com). Parts of the code were
initially ported from the Python
[autograd](https://github.com/HIPS/autograd) package.  I'd like to
thank autograd author Dougal Maclaurin for his support.  See [(Baydin
et al. 2015)](https://arxiv.org/abs/1502.05767) for a general review
of automatic differentiation, [autograd
tutorial](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md)
for some Python examples, and Dougal's PhD thesis for design
principles.  [JuliaDiff](http://www.juliadiff.org/) and
[FluxML](https://github.com/FluxML) have alternative differentiation
tools for Julia.  I would like to thank Carlo Lucibello, Mike Innes,
Rene Donner, Ekin Akyurek, Ozan Arkan Can and Emre Yolcu for their
contributions.

The suggested citation for AutoGrad is:

```
@inproceedings{knet2016mlsys,
  author={Yuret, Deniz},
  title={Knet: beginning deep learning with 100 lines of Julia},
  year={2016},
  booktitle={Machine Learning Systems Workshop at NIPS 2016}
}
```
