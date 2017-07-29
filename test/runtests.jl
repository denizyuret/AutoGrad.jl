# Uncomment these if you want lots of messages:
# import Base.Test: default_handler, Success, Failure, Error
# default_handler(r::Success) = info("$(r.expr)")
# default_handler(r::Failure) = warn("$(r.expr) FAILED")
# default_handler(r::Error)   = warn("$(r.expr): $(r.err)")

include("interfaces.jl")
include("indexing.jl")
include("rosenbrock.jl")
include("highorder.jl")
include("neuralnet.jl")
if VERSION < v"0.6-" # TODO for v0.6
include("primitives.jl")
end
