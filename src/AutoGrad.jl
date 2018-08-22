module AutoGrad
using LinearAlgebra, Statistics, SpecialFunctions

# Use AutoGrad.dir(path...) to construct paths relative to AutoGrad root.
dir(path...) = joinpath(dirname(dirname(@__FILE__)),path...)

# Uncomment include("../util/debug.jl") and the following macro line to see debug output
# macro dbg(x); esc(:(println(_dbg($x)))); end;
macro dbg(x); end

# To perform profiling of AutoGrad internals, uncomment the following lines. Make sure to Pkg.add("TimerOutputs").
# using TimerOutputs
# macro prof(label,ex); :(@timeit $(esc(label)) $(esc(ex))); end
macro prof(label,ex); esc(ex); end

export grad, gradloss, getval
export @primitive, @zerograd, @primitive1, @zerograd1

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

# Uncomment these for debugging:
# include("../util/debug.jl")
include("../test/gradcheck.jl")

end # module
