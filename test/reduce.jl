@testset "reduce" begin
    a = grad(x->mean(mean(x,(1,2))))(rand(2,2,2,2))
    @test all(a .== 0.0625)
end