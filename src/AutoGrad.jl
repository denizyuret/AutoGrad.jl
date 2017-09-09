VERSION >= v"0.4.0-dev+6521" && __precompile__()

module AutoGrad
using Compat

# utilities for debugging and profiling.
macro dbg(i,x); if i & 0 != 0; esc(:(println(_dbg($x)))); end; end;
macro gs(); if false; esc(:(ccall(("cudaDeviceSynchronize","libcudart"),UInt32,()))); end; end

importall Base  # defining getindex, sin, etc.
export grad, gradloss, check_grads, gradcheck, gradcheckN, getval
export @primitive, @zerograd, recorder, Rec, Grad  # the last three are required for the macros to work
datapath = joinpath(dirname(@__FILE__),"..","data")

include("core.jl")
include("unfuse.jl")
include("dotfunc.jl")
include("gradcheck.jl")
include("util.jl")
include("interfaces.jl")
include("base/reduce.jl")
include("base/number.jl")
include("base/float.jl")
include("base/broadcast.jl")
include("base/math.jl")
include("base/abstractarray.jl")
include("base/abstractarraymath.jl")
include("base/arraymath.jl")
include("linalg/matmul.jl")
include("linalg/dense.jl")
include("linalg/generic.jl")
include("special/trig.jl")
if VERSION < v"0.6.0"
    include("special/bessel.jl") ### Removed from Base in Julia6
    include("special/erf.jl")    ### Removed from Base in Julia6
    include("special/gamma.jl")  ### Removed from Base in Julia6
end

end # module
