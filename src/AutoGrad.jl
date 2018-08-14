# __precompile__() # is now the default
module AutoGrad
using LinearAlgebra, Statistics, SpecialFunctions

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

export grad, gradloss, getval
export @primitive, @zerograd, @primitive1, @zerograd1, recorder, Rec, Grad  # the last three are required for the macros to work
dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

include("core.jl")
include("broadcast.jl")
include("macros.jl")
include("base.jl")
include("math.jl")
include("statistics.jl")
include("linearalgebra.jl")
include("specialfunctions.jl")
include("getindex.jl")
include("iterate.jl")
include("cat.jl")
# include("../test/gradcheck.jl"); export gradcheck, randcheck, check_grads, gradcheckN # TODO: remove this before release

end # module
