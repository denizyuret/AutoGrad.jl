module AutoGrad
export Param, params, grad, value, @diff
export gradloss, getval
export @primitive, @zerograd, @primitive1, @zerograd1

# Set ENV["AUTOGRAD_TIMER"]="true" and Pkg.build("AutoGrad") if you want profiling information in AutoGrad.to
using TimerOutputs, Libdl
const TIMER=haskey(ENV,"AUTOGRAD_TIMER")
const to = TimerOutput()
macro gs(); if !isempty(Libdl.find_library("libcudart")); esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))); end; end
macro timer(name,expr); TIMER ? :(@timeit to $(esc(name)) (a=$(esc(expr));@gs;a)) : esc(expr); end

"""
Usage:

    x = Param([1,2,3])          # user declares parameters
    x => P([1,2,3])             # they are wrapped in a struct
    value(x) => [1,2,3]         # we can get the original value
    sum(abs2,x) => 14           # they act like regular values outside of differentiation
    y = @diff sum(abs2,x)       # if you want the gradients
    y => T(14)                  # you get another struct
    value(y) => 14              # which represents the same value
    grad(y,x) => [2,4,6]        # but also contains gradients for all Params
    
`Param(x)` returns a struct that acts like `x` but marks it as a parameter you want to
compute gradients with respect to.

`@diff expr` evaluates an expression and returns a struct that contains its value (which
should be a scalar) and gradient information.

`grad(y, x)` returns the gradient of `y` (output by @diff) with respect to any parameter
`x::Param`, or  `nothing` if the gradient is 0.

`value(x)` returns the value associated with `x` if `x` is a `Param` or the output of
`@diff`, otherwise returns `x`.

`params(x)` returns an array of Params found by a recursive search of object `x`.

Alternative usage:

    x = [1 2 3]
    f(x) = sum(abs2, x)
    f(x) => 14
    grad(f)(x) => [2 4 6]
    gradloss(f)(x) => ([2 4 6], 14)

Given a scalar valued function `f`, `grad(f,argnum=1)` returns another function `g` which
takes the same inputs as `f` and returns the gradient of the output with respect to the
argnum'th argument. `gradloss` is similar except the resulting function also returns f's
output.
"""
AutoGrad, Param, params, :(@diff), grad, value, gradloss

using LinearAlgebra, Statistics, SpecialFunctions
include("core.jl")
#include("broadcast.jl")
include("macros.jl")
include("getindex.jl")
include("iterate.jl")
include("sum_outgrads.jl")
include("base.jl")
include("math.jl")
include("statistics.jl")
include("linearalgebra.jl")
include("specialfunctions.jl")
include("cat.jl")
include("show.jl")
include("params.jl")
include("../test/gradcheck.jl")

"Use `AutoGrad.dir(path...)` to construct paths relative to AutoGrad root."
dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

# Deprecations:
@deprecate Rec Value
@deprecate getval value

end # module
