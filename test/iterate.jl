include("header.jl")

@testset "iterate" begin
    w = [randn(2,3,5),randn(1,3,5),randn(2,2,5),randn(2,3,2)]
    wdict = Dict(:w1=>randn(2,3),:w2=>randn(3,2))

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

    function itr3(w)
        total = 0.0
        for (k,wi) in w; total+=sum(wi); end
        return total
    end
    

    @test gradcheck(itr1,w)
    @test gradcheck(itr2,w)
    @test gradcheck(itr1,w[1])
    @test gradcheck(itr2,w[1])
    @test gradcheck(itr3,wdict) #fails

end
