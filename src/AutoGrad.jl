module AutoGrad

using Compat
importall Base  # defining getindex, sin, etc.
# importall Knet  # quadloss etc.
export grad, @primitive, D1, D2, Dn, check_grads

# macro dbgcore(x); esc(:(println($x))); end
# macro dbgutil(x); esc(:(println($x))); end
macro dbgcore(x); end
macro dbgutil(x); end

include("core.jl")
include("util.jl")
include("collections.jl")
include("base/abstractarray.jl")
include("base/arraymath.jl")
include("base/number.jl")
include("base/float.jl")
include("base/floatfuncs.jl")
include("base/complex.jl")
include("base/broadcast.jl")
include("base/math.jl")
include("base/reduce.jl")
include("linalg/matmul.jl")
include("linalg/dense.jl")
include("linalg/generic.jl")
include("special/bessel.jl")
include("special/erf.jl")
include("special/gamma.jl")
include("special/trig.jl")
:ok

end # module
