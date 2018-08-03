# Uncomment these if you want lots of messages:
# import Base.Test: default_handler, Success, Failure, Error
# default_handler(r::Success) = info("$(r.expr)")
# default_handler(r::Failure) = warn("$(r.expr) FAILED")
# default_handler(r::Error)   = warn("$(r.expr): $(r.err)")

include("interfaces.jl")
include("indexing.jl")
include("rosenbrock.jl")
include("highorder.jl")         # fails because of undefined sin
include("neuralnet.jl")         # problem with sum
include("primitives.jl")        # cannot convert ungetindex2 to scalar function, applyN is broken with broadcast
