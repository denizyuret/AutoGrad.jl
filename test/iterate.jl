include("header.jl")
w = [randn(2,3,5),randn(1,3,5),randn(2,2,5),randn(2,3,2)]

function itr1(w)
    total = 0.0
    for wi in w; total+=sum(wi); end
    return total
end

function itr2(w)
    total = 0.0
    for (i,wi) in enumerate(w); total+=sum(wi); end
    return total
end

@testset "iterate" begin
    @test gradcheckN(itr1,w)
    @test gradcheckN(itr2,w)
    @test gradcheckN(itr1,w[1])
    @test gradcheckN(itr2,w[1])
end
