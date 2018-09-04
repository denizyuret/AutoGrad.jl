# AutoGrad

<!--
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_0.6.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_0.7.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_1.0.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
-->

[![Build Status](https://travis-ci.org/denizyuret/AutoGrad.jl.svg?branch=master)](https://travis-ci.org/denizyuret/AutoGrad.jl)
[![coveralls](https://coveralls.io/repos/github/denizyuret/AutoGrad.jl/badge.svg?branch=master)](https://coveralls.io/github/denizyuret/AutoGrad.jl?branch=master)
[![codecov](https://codecov.io/gh/denizyuret/AutoGrad.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/denizyuret/AutoGrad.jl)

AutoGrad.jl is an automatic differentiation package for Julia.  It
started as a port of the popular Python
[autograd](https://github.com/HIPS/autograd) package and forms the
foundation of the [Knet](https://github.com/denizyuret/Knet.jl) Julia
deep learning framework.  AutoGrad can differentiate regular Julia
code that includes loops, conditionals, helper functions, closures
etc. by keeping track of the primitive operations and using this
execution trace to compute gradients.  It uses reverse mode
differentiation (a.k.a. backpropagation) so it can efficiently handle
functions with large array inputs and scalar outputs.  It can compute
gradients of gradients to handle higher order derivatives.

## Installation

You can install AutoGrad in Julia using:
```
julia> using Pkg; Pkg.add("AutoGrad")
```

In order to use it in your code start with:
```
using AutoGrad
```

## Interface

```
x = Param([1,2,3])		# user declares parameters
x => P([1,2,3])			# they are wrapped in a struct
value(x) => [1,2,3]		# we can get the original value
sum(abs2,x) => 14		# they act like regular values outside of differentiation
y = @diff sum(abs2,x)	        # if you want the gradients
y => T(14)			# you get another struct
value(y) => 14			# which represents the same value
grad(y,x) => [2,4,6]	        # but also contains gradients for all Params
```

## Old Interface

Pre v1.1 AutoGrad only supported the following `grad` interface. This
is still supported.

```
x = [1,2,3]
f(x) = sum(abs2,x)
g = grad(f)
f(x) => 14
g(x) => [2,4,6]
```

## Example

Here is a linear regression example using [callable objects](https://docs.julialang.org/en/stable/manual/methods/#Function-like-objects-1):

```
struct Linear; w; b; end		# user defines a model
(f::Linear)(x) = (f.w * x .+ f.b)

# Initialize a model as a callable object with parameters:
f = Linear(Param(randn(10,100), Param(randn(10))))

# SGD training loop:
for (x,y) in data
    loss = @diff sum(abs2,f(x)-y)
    for w in params(f)
        g = grad(loss,w)
	axpy!(-0.01, g, w)
    end
end
```

See the [examples
directory](https://github.com/denizyuret/AutoGrad.jl/blob/master/examples)
for more examples.

## Extending AutoGrad

AutoGrad can only handle a function if the primitives it uses have
known gradients.  You can add your own primitives with gradients using
the `@primitive` and `@zerograd` macros in
[macros.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/macros.jl)
Here is an example:

```
@primitive log(x),dy,y  (dy .* (1 ./ x))
```

The `@primitive` macro marks the `log(::Any)` method as a new
primitive and the next expression defines a gradient function wrt the
first argument.  The gradient expressions can refer to the
parameter(s) `x`, the return variable `y` and its gradient `dy`
(optionally indicated after the argument list) in the method
declaration. For functions with multiple inputs multiple gradient
expressions may be given. Non-existent or zero gradients can be
specified by omitting a gradient expression or using `nothing` in
place of one. By default the broadcasting version `log.(x)` is also
defined as a primitive, use the `@primitive1` macro if you don't want
this.

Note that Julia supports multiple-dispatch, i.e. a function may have
multiple methods each supporting different argument types.  For
example `log(::Float32)` and `log(::BigFloat)` are two different log
methods.  In AutoGrad.jl each method can be defined independently as a
primitive and can have its own specific gradient. Generally AutoGrad
defines gradients without using argument types to keep the rules
generic.

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
tools for Julia.  I would like to thank the current contributors:

* Carlo Lucibello
* Ekin Akyürek
* Emre Yolcu
* Jarrett Revels
* Mike Innes
* Ozan Arkan Can
* Rene Donner

The suggested citation for AutoGrad is:

```
@inproceedings{knet2016mlsys,
  author={Yuret, Deniz},
  title={Knet: beginning deep learning with 100 lines of Julia},
  year={2016},
  booktitle={Machine Learning Systems Workshop at NIPS 2016}
}
```
