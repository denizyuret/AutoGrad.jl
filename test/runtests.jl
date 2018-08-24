using Test

@testset "AutoGrad" begin
@time include("base.jl")
#TODO include("broadcast.jl")
@time include("cat.jl")
@time include("core.jl")
##@time include("getindex.jl")
@time include("iterate.jl")
@time include("linearalgebra.jl")
#TODO include("macros.jl")
@time include("math.jl")
@time include("specialfunctions.jl")
@time include("statistics.jl")

@time include("rosenbrock.jl")
@time include("highorder.jl")
@time include("neuralnet.jl")
end
