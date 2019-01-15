module AutoGrad
export Param, params, grad, value, @diff
export gradloss, getval
export @primitive, @zerograd, @primitive1, @zerograd1, @primitive2, @zerograd2

# Set ENV["AUTOGRAD_TIMER"]="true" and Pkg.build("AutoGrad") if you want profiling information in AutoGrad.to
using TimerOutputs, Libdl
const TIMER=haskey(ENV,"AUTOGRAD_TIMER")
const to = TimerOutput()
macro gs(); if !isempty(Libdl.find_library("libcudart")); esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))); end; end
macro timer(name,expr); TIMER ? :(@timeit to $(esc(name)) (a=$(esc(expr));@gs;a)) : esc(expr); end

"""
Usage:

    x = Param([1,2,3])          # user declares parameters with `Param`
    x => P([1,2,3])             # `Param` is just a struct wrapping a value
    value(x) => [1,2,3]         # `value` returns the thing wrapped
    sum(x .* x) => 14           # Params act like regular values
    y = @diff sum(x .* x)       # Except when we differentiate using `@diff`
    y => T(14)                  # you get another struct
    value(y) => 14              # which carries the same result
    params(y) => [x]            # and the Params that it depends on 
    grad(y,x) => [2,4,6]        # and the gradients for all Params
    
`Param(x)` returns a struct that acts like `x` but marks it as a parameter you want to
compute gradients with respect to.

`@diff expr` evaluates an expression and returns a struct that contains the result (which
should be a scalar) and gradient information.

`grad(y, x)` returns the gradient of `y` (output by @diff) with respect to any parameter
`x::Param`, or  `nothing` if the gradient is 0.

`value(x)` returns the value associated with `x` if `x` is a `Param` or the output of
`@diff`, otherwise returns `x`.

`params(x)` returns an iterator of Params found by a recursive search of object `x`.

Alternative usage:

    x = [1 2 3]
    f(x) = sum(x .* x)
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
include("macros.jl")
include("getindex.jl")
include("iterate.jl")
include("sum_outgrads.jl")
include("unbroadcast.jl")
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
