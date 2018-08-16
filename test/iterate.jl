include("header.jl")

@testset "iterate" begin
    d = Dict(:a=>1, :b=>2)
    @test grad(x->sum(values(x)))(d) == Dict(:a=>1, :b=>1)
end