include("header.jl")

@testset "rosenbrock" begin
    # info("Test rosenbrock with map...")
    a1 = rand(3)                    # TODO: FAIL: some high-order (b1sum) tests with rand(2)
    t1 = (a1...,)
    d1 = Dict(); for i=1:length(a1); d1[i]=a1[i]; end
    b0(x) = sum(map((i, j) -> (1 - j)^2 + 100*(i - j^2)^2, x[2:end], x[1:end-1]))
    b1 = grad(b0)
    b1sum(x)=AutoGrad.sumvalues(b1(x))
    @test check_grads(b0,a1)
    @test check_grads(b0,t1)
    @test check_grads(b1sum,a1) # TODO: fail with size 2
    @test check_grads(b1sum,t1) # TODO: fail with size 2
    @time b1sum(rand(10000))
end

nothing
