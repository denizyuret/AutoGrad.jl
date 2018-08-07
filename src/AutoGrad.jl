# __precompile__() # is now the default
module AutoGrad
using Requires, LinearAlgebra, Statistics

# To see debug output of AutoGrad internals, set DBGFLAGS to
# non-zero. Each bit of DBGFLAGS can be used to show a subset of dbg
# messages indicated by the `bit` argument to the `dbg` macro.
const DBGFLAGS = 0
macro dbg(bit,x); if (1<<bit) & DBGFLAGS != 0; esc(:(println(_dbg($x)))); end; end;

# To perform profiling of AutoGrad internals, set PROFILING to
# true. Make sure to Pkg.add("TimerOutputs").
const PROFILING = false
if PROFILING
    eval(Expr(:using,:TimerOutputs))
    macro prof(label,ex); :(@timeit $(esc(label)) $(esc(ex))); end
else
    macro prof(label,ex); esc(ex); end
end

#TODO importall Base  # defining getindex, sin, etc.
export grad, gradloss, check_grads, gradcheck, gradcheckN, getval
export @primitive, @zerograd, @primitive2, @zerograd2, recorder, Rec, Grad  # the last three are required for the macros to work
datapath = joinpath(dirname(@__FILE__),"..","data")

include("core.jl")
include("broadcast.jl")
include("macros.jl")
include("base.jl")
include("math.jl")
include("statistics.jl")
include("linearalgebra.jl")
include("getindex.jl")
include("iterate.jl")
include("cat.jl")

function __init__()
    @require SpecialFunctions="276daf66-3868-5448-9aa4-cd146d93841b" include("specialfunctions.jl")
end


#include("gradcheck.jl")
#include("debug.jl")
# #include("base/reduce.jl")  # figure out sumabs
# include("base/number.jl")
# include("base/float.jl")
# # include("base/broadcast.jl")  # remove duplicates
# include("base/math.jl")
# include("base/abstractarray.jl")
# include("base/abstractarraymath.jl")
# include("base/arraymath.jl")
# # include("base/statistics.jl") # deprecated var, std, import Statistics
# include("linalg/matmul.jl")  # dot moved to LinearAlgebra
# include("linalg/dense.jl")
# include("linalg/generic.jl")
# include("special/trig.jl")
# if Pkg.installed("SpecialFunctions") != nothing
#     eval(Expr(:using,:SpecialFunctions))
#     include("special/bessel.jl")
#     include("special/erf.jl")
#     include("special/gamma.jl")
# end

end # module
