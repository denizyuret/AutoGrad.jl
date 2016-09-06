VERSION >= v"0.4.0-dev+6521" && __precompile__()

module AutoGrad

importall Base  # defining getindex, sin, etc.
export grad, check_grads, @primitive, @zerograd
export Value, getval, recorder, Grad, fixdomain  # these are required for the macros to work

include("core.jl")
include("util.jl")
include("interfaces.jl")
include("base/reduce.jl")
include("base/number.jl")
include("base/float.jl")
include("base/broadcast.jl")
include("base/math.jl")
include("base/abstractarray.jl")
include("base/arraymath.jl")
include("linalg/matmul.jl")
include("linalg/dense.jl")
include("linalg/generic.jl")
include("special/bessel.jl")
include("special/erf.jl")
include("special/gamma.jl")
include("special/trig.jl")

end # module
