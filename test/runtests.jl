# Uncomment these if you want lots of messages:
# import Base.Test: default_handler, Success, Failure, Error
# default_handler(r::Success) = info("$(r.expr)")
# default_handler(r::Failure) = warn("$(r.expr) FAILED")
# default_handler(r::Error)   = warn("$(r.expr): $(r.err)")

#include("core.jl")
#include("broadcast.jl")
#include("macros.jl")
#include("base.jl")
@time include("math.jl")
@time include("statistics.jl")
#include("linearalgebra.jl")
@time include("specialfunctions.jl")
@time include("getindex.jl")
@time include("iterate.jl")
#include("cat.jl")
@time include("rosenbrock.jl")
@time include("highorder.jl") 
@time include("neuralnet.jl") 
