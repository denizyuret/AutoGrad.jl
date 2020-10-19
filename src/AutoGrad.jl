module AutoGrad
export Param, params, grad, value, @diff
export gradloss, getval
export @primitive, @zerograd, @primitive1, @zerograd1, @primitive2, @zerograd2
export @gcheck

# Set ENV["AUTOGRAD_TIMER"]="true" and Pkg.build("AutoGrad") if you want profiling information in AutoGrad.to
using TimerOutputs, Libdl
const TIMER=haskey(ENV,"AUTOGRAD_TIMER")
const to = TimerOutput()
macro gs(); if !isempty(Libdl.find_library("libcudart")); esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))); end; end
macro timer(name,expr); TIMER ? :(@timeit to $(esc(name)) (a=$(esc(expr));@gs;a)) : esc(expr); end

"""
Usage:

    x = Param([1,2,3])          # The user declares parameters with `Param`
    y = @diff sum(x .* x)       # computes gradients using `@diff`
    grad(y,x) => [2,4,6]        # looks up the gradient of a parameter with `grad`

`Param(x)` returns a struct that acts like `x` but marks it as a parameter you want to
compute gradients with respect to.

`@diff expr` evaluates an expression and returns a struct that contains its value (which
should be a scalar) and gradients with respect to the `Param`s used in the computation.

`grad(y, x)` returns the gradient of a `@diff` result `y` with respect to any parameter
`x::Param`. (`nothing` may be returned if the gradient is 0).

`value(x)` returns the value associated with `x` if `x` is a `Param` or the output of
`@diff`, otherwise returns `x`.

`params(x)` returns an iterator of `Param`s found by a recursive search of object `x`, which
is typically a model or a `@diff` result.

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
include("sparse.jl")
include("getindex.jl")
include("iterate.jl")
include("addto.jl")
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
