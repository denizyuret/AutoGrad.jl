module AutoGrad

using Compat
importall Base  # defining getindex, sin, etc.
export grad, check_grads, @primitive, @zerograd

_dbg(x)=x
# macro dbgcore(x); esc(:(println(_dbg($x)))); end
# macro dbgutil(x); esc(:(println(_dbg($x)))); end
macro dbgcore(x); end
macro dbgutil(x); end

include("core.jl")
include("util.jl")
include("interfaces.jl")
include("base/abstractarray.jl")
include("base/arraymath.jl")
include("base/broadcast.jl")
include("base/float.jl")
include("base/math.jl")
include("base/number.jl")
include("base/reduce.jl")
include("linalg/matmul.jl")
# include("linalg/dense.jl")
# include("linalg/generic.jl")
# include("special/bessel.jl")
# include("special/erf.jl")
# include("special/gamma.jl")
# include("special/trig.jl")

end # module
