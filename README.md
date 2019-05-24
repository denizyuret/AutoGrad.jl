# AutoGrad

<!--
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_0.6.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_0.7.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
[![AutoGrad](http://pkg.julialang.org/badges/AutoGrad_1.0.svg)](http://pkg.julialang.org/?pkg=AutoGrad)
-->

[![Build Status](https://travis-ci.org/denizyuret/AutoGrad.jl.svg?branch=master)](https://travis-ci.org/denizyuret/AutoGrad.jl)
[![coveralls](https://coveralls.io/repos/github/denizyuret/AutoGrad.jl/badge.svg?branch=master)](https://coveralls.io/github/denizyuret/AutoGrad.jl?branch=master)
[![codecov](https://codecov.io/gh/denizyuret/AutoGrad.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/denizyuret/AutoGrad.jl)

AutoGrad.jl is an automatic differentiation package for Julia.  It started as a port of the
popular Python [autograd](https://github.com/HIPS/autograd) package and forms the foundation
of the [Knet](https://github.com/denizyuret/Knet.jl) Julia deep learning framework.
AutoGrad can differentiate regular Julia code that includes loops, conditionals, helper
functions, closures etc. by keeping track of the primitive operations and using this
execution trace to compute gradients.  It uses reverse mode differentiation
(a.k.a. backpropagation) so it can efficiently handle functions with large array inputs and
scalar outputs.  It can compute gradients of gradients to handle higher order derivatives.

## Installation

You can install AutoGrad in Julia using:
```julia
julia> using Pkg; Pkg.add("AutoGrad")
```

In order to use it in your code start with:
```julia
using AutoGrad
```

## Interface

```julia
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

Pre v1.1 AutoGrad only supported the following `grad` interface. This is still supported.

```julia
x = [1,2,3]
f(x) = sum(abs2,x)
g = grad(f)
f(x) => 14
g(x) => [2,4,6]
```

## Example

Here is a linear regression example using [callable objects](https://docs.julialang.org/en/stable/manual/methods/#Function-like-objects-1):

```julia
struct Linear; w; b; end		# user defines a model
(f::Linear)(x) = (f.w * x .+ f.b)

# Initialize a model as a callable object with parameters:
f = Linear(Param(randn(10,100)), Param(randn(10)))

# SGD training loop:
for (x,y) in data
    loss = @diff sum(abs2,f(x)-y)
    for w in params(f)
        g = grad(loss,w)
	axpy!(-0.01, g, w)
    end
end
```

See the [examples directory](https://github.com/denizyuret/AutoGrad.jl/blob/master/examples)
for more examples.

## Extending AutoGrad

AutoGrad can only handle a function if the primitives it uses have known gradients.  You can
add your own primitives with gradients using the `@primitive` and `@zerograd` macros in
[macros.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/macros.jl) Here is an
example:

```julia
@primitive log(x),dy,y  (dy .* (1 ./ x))
```

The `@primitive` macro marks the `log(::Any)` method as a new primitive and the next
expression defines a gradient function wrt the first argument.  The gradient expressions can
refer to the parameter(s) `x`, the return variable `y` and its gradient `dy` (optionally
indicated after the argument list) in the method declaration. For functions with multiple
inputs multiple gradient expressions may be given. Non-existent or zero gradients can be
specified by omitting a gradient expression or using `nothing` in place of one. By default
the broadcasting version `log.(x)` is also defined as a primitive, use the `@primitive1`
macro if you don't want this.

Note that Julia supports multiple-dispatch, i.e. a function may have multiple methods each
supporting different argument types.  For example `log(::Float32)` and `log(::BigFloat)` are
two different log methods.  In AutoGrad.jl each method can be defined independently as a
primitive and can have its own specific gradient. Generally AutoGrad defines gradients
without using argument types to keep the rules generic.

## Debugging and Profiling

To view the contents of the computational graph after differentiating a function you can use
the following:

```julia
julia> AutoGrad.gcnode(::AutoGrad.Node)=nothing  # without this some values may be lost
julia> w = Param(rand(2,3)); b = Param(rand(2,1)); x = rand(3,4); y = rand(2,4);
julia> J = @diff sum(abs2, w*x .+ b - y)
T(14.695603907991153)
julia> [J]  # displaying J in an Array causes pretty printing
1. P(Array{Float64,2}(2,3)) ∇=Array{Float64,2}(2,3)
2. Array{Float64,2}(2,4) = *(Array{Float64,2}(2,3), Array{Float64,2}(3,4))) ∇=Array{Float64,2}(2,4)
3. P(Array{Float64,2}(2,1)) ∇=Array{Float64,2}(2,1)
4. Array{Float64,2}(2,4) = broadcast(+, Array{Float64,2}(2,4), Array{Float64,2}(2,1))) ∇=Array{Float64,2}(2,4)
5. Array{Float64,2}(2,4) = -(Array{Float64,2}(2,4), Array{Float64,2}(2,4))) ∇=Array{Float64,2}(2,4)
6. 14.695603907991153 = sum(abs2, Array{Float64,2}(2,4))) ∇=1.0
julia> z = collect(J.list)  # collect creates a Node array with reverse order
julia> dump(z[5], maxdepth=1)  # allowing you to look at individual Nodes and Values
AutoGrad.Node
  Value: AutoGrad.Result{Array{Float64,2}}
  parents: Array{AutoGrad.Node}((2,))
  children: Array{AutoGrad.Node}((1,))
  outgrad: Array{Float64}((2, 4)) [3.82753 2.19124 3.26769 3.0075; 2.81565 2.3903 1.84373 1.60228]
  cdr: AutoGrad.Node
julia> dump(z[5].Value, maxdepth=2)
AutoGrad.Result{Array{Float64,2}}
  value: Array{Float64}((2, 4)) [1.16724 1.07224 0.935047 0.895262; 0.687182 0.589704 0.517114 0.495718]
  func: * (function of type typeof(*))
  args: Tuple{Param{Array{Float64,2}},Array{Float64,2}}
    1: Param{Array{Float64,2}}
    2: Array{Float64}((3, 4)) [0.515282 0.257471 0.140791 0.127632; 0.705288 0.783289 0.361965 0.311965; 0.780549 0.691645 0.853317 0.843374]
  kwargs: Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}}
    data: NamedTuple{(),Tuple{}} NamedTuple()
    itr: Tuple{} ()
```

To profile AutoGrad using TimerOutputs.jl, set the environment variable
`ENV["AUTOGRAD_TIMER"]="true"` and rebuild AutoGrad with `Pkg.build("AutoGrad")`, before
evaluating `using AutoGrad`. The environment variable `AUTOGRAD_TIMER` is only checked at
compile time, not at run time for performance reasons. This will collect detailed timing
information but slows the code down, when you are done don't forget to
`delete!(ENV,"AUTOGRAD_TIMER")` and rebuild AutoGrad. In the example below, the symbol `sum`
indicates the time spent on the forward pass of the `sum` function and `sum[2]` indicates
the time spent on the backward pass for the second argument. `record` and `sum_outgrads` are
functions internal to AutoGrad.

```julia
julia> ENV["AUTOGRAD_TIMER"]="true"
julia> using Pkg; Pkg.build("AutoGrad")
julia> using AutoGrad, TimerOutputs
julia> reset_timer!(AutoGrad.to)
julia> w = Param(rand(2,3)); b = Param(rand(2,1)); x = rand(3,4); y = rand(2,4);
julia> J = @diff sum(abs2, w*x .+ b - y)
julia> AutoGrad.to
───────────────────────────────────────────────────────────────────────
                                Time                   Allocations      
                        ──────────────────────   ───────────────────────
    Tot / % measured:        4.62s / 30.4%            546MiB / 25.0%    

 Section        ncalls     time   %tot     avg     alloc   %tot      avg
 ───────────────────────────────────────────────────────────────────────
 +.[2]               1    328ms  23.3%   328ms   46.4MiB  34.1%  46.4MiB
 sum[2]              1    288ms  20.5%   288ms   40.0MiB  29.4%  40.0MiB
   *                 1   38.8ms  2.76%  38.8ms    595KiB  0.43%   595KiB
 *                   1    269ms  19.2%   269ms    955KiB  0.68%   955KiB
 +.                  1    139ms  9.92%   139ms   20.4MiB  15.0%  20.4MiB
 *[1]                1    117ms  8.33%   117ms   9.41MiB  6.90%  9.41MiB
 record              4   88.7ms  6.31%  22.2ms   3.49MiB  2.56%   894KiB
 -[1]                1   65.9ms  4.69%  65.9ms   10.0MiB  7.32%  10.0MiB
 -                   1   55.8ms  3.97%  55.8ms    929KiB  0.67%   929KiB
 sum                 1   50.0ms  3.56%  50.0ms   4.68MiB  3.44%  4.68MiB
 +.[1]               1   1.78ms  0.13%  1.78ms   37.7KiB  0.03%  37.7KiB
 sum_outgrads        5   1.41ms  0.10%   282μs   28.2KiB  0.02%  5.64KiB
 ───────────────────────────────────────────────────────────────────────
```

## Code structure

[core.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/core.jl) implements the
main functionality and acts as the main documentation source.
[macros.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/macros.jl) has some
support functions to define and test new primitives.
[getindex.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/getindex.jl),
[iterate.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/iterate.jl) and
[cat.jl](https://github.com/denizyuret/AutoGrad.jl/blob/master/src/cat.jl) set up support
for common data structures including Arrays, Tuples, and Dictionaries.  The numerical
gradients are defined in files such as `base.jl` and `math.jl`.

## Current status and future work

The gradient coverage and unit testing are spotty, I am still adding more gradients and
tests to cover the Julia base. Documentation needs to be improved. Overwriting functions
(e.g. `setindex!`) are not supported. Efficiency could be improved by reducing runtime
compilation, memoization, and support for static computation.

## Acknowledgments and references

AutoGrad.jl was written by [Deniz Yuret](http://www.denizyuret.com). Parts of the code were
initially ported from the Python [autograd](https://github.com/HIPS/autograd) package.  I'd
like to thank autograd author Dougal Maclaurin for his support.  See [(Baydin et
al. 2015)](https://arxiv.org/abs/1502.05767) for a general review of automatic
differentiation, [autograd
tutorial](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md) for some Python
examples, and Dougal's PhD thesis for design principles.
[JuliaDiff](http://www.juliadiff.org/) and [FluxML](https://github.com/FluxML) have
alternative differentiation tools for Julia.  I would like to thank the current
contributors:

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
