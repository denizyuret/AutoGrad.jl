# AutoGrad

[![Build Status](https://travis-ci.org/denizyuret/AutoGrad.jl.svg?branch=master)](https://travis-ci.org/denizyuret/AutoGrad.jl)
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_0.4.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_0.5.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
<!-- 
TODO: https://github.com/JuliaCI/Coverage.jl
[![Coverage Status](https://coveralls.io/repos/denizyuret/AutoGrad.jl/badge.svg)](https://coveralls.io/r/denizyuret/AutoGrad.jl)
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_0.3.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
-->

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
julia> Pkg.add("AutoGrad")
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

## Extending AutoGrad

AutoGrad can only handle a function if the primitives it uses have
known gradients.  You can add your own primitives with gradients as
described in detail in
[core.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/core.jl)
or using the `@primitive` and `@zerograd` macros in
[util.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/util.jl)
Here is an example:

```
@primitive hypot(x1::Number,x2::Number),dy,y  (dy->dy*x1/y)  (dy->dy*x2/y)
```

The `@primitive` macro marks the `hypot(::Number,::Number)` method as
a new primitive and the next two expressions define gradient functions
wrt the first and second argument.  The gradient expressions can refer
to the parameters `(x1,x2)`, the return variable `y` and its gradient
`dy` (optionally indicated after the argument list) in the method
declaration.

Note that Julia supports multiple-dispatch, i.e. a function may have
multiple methods each supporting different argument types.  For
example `hypot(x1::Array,x2::Array)` is another hypot method.  In
AutoGrad.jl each method can independently be defined as a primitive
and can have its own specific gradient.

## Code structure

[core.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/core.jl)
implements the main functionality and acts as the main documentation
source.
[util.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/util.jl)
has some support functions to define and test new primitives.
[interfaces.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/interfaces.jl)
sets up support for common data structures including Arrays, Tuples,
and Dictionaries.  The numerical gradients are defined in files such
as `base/math.jl`, `special/trig.jl` that mirror the organization
under `julia/base`.

## Current status and future work

The gradient coverage is spotty, I am still adding more gradients to
cover the Julia base.  Next steps are to make models faster by
providing support for GPU operations and overwriting functions (to
avoid memory allocation).  I should also find out about the efficiency
of closures and untyped functions in Julia which are used extensively
in the code.

## Acknowledgments and references

AutoGrad.jl was written by [Deniz
Yuret](http://www.denizyuret.com). Large parts of the code are
directly ported from the Python
[autograd](https://github.com/HIPS/autograd) package.  I'd like to
thank autograd author Dougal Maclaurin for his support.  See [(Baydin
et al. 2015)](https://arxiv.org/abs/1502.05767) for a general review
of automatic differentiation, [autograd
tutorial](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md)
for some Python examples, and Dougal's PhD thesis for design
principles.  [JuliaDiff](http://www.juliadiff.org/) has alternative
differentiation tools for Julia.  I would like to thank my students
Ozan Arkan Can and Emre Yolcu for helpful contributions.

The suggested citation for AutoGrad is:

```
@misc{autograd,
  author={Yuret, Deniz},
  title={Autograd: an automatic differentiation package for Julia},
  year={2016},
  howpublished={\url{https://github.com/denizyuret/AutoGrad.jl}}
}
```
