module AutoGrad

using LinearAlgebra, Statistics, SpecialFunctions
export Param, differentiate, gradient, value, grad, gradloss, getval
export @primitive, @zerograd, @primitive1, @zerograd1

include("core.jl")
include("broadcast.jl")
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
include("../test/gradcheck.jl")

"Use `AutoGrad.dir(path...)` to construct paths relative to AutoGrad root."
dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

"Use `using AutoGrad.Abbrev` to get the shortcuts gr, df, pa."
module Abbrev
using ..AutoGrad
const gr = gradient
const df = differentiate
const pa = Param
export gr,df,pa
end

# Deprecations:
@deprecate Rec Value
@deprecate getval value

end # module
