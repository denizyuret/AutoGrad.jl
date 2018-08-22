include("header.jl")

@testset "rosenbrock" begin
    # info("Test rosenbrock with map...")
    b0(x) = sum(map((i, j) -> (1 - j)^2 + 100*(i - j^2)^2, x[2:end], x[1:end-1]))
    b1 = grad(b0)
    b1sum(x)=sum(b1(x))
    @test gradcheck(b0,rand(2))
    @test gradcheck(b0,(rand(2)...,))
    @test gradcheck(b0,rand(3))
    @test gradcheck(b0,(rand(3)...,))
    @test gradcheck(b1sum,rand(2))
    @test gradcheck(b1sum,(rand(2)...,))
    @test gradcheck(b1sum,rand(3))
    @test gradcheck(b1sum,(rand(3)...,))
    @time b1sum(rand(10000))
end

nothing
