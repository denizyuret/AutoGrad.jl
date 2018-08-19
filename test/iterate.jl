include("header.jl")

@testset "iterate" begin
    warray  = [randn(2,3,5),randn(1,3,5)]
    wtuple  = (randn(2,3,5),randn(1,3,5))
    wdict   = Dict(:w1=>randn(2,3,5),:w2=>randn(1,3,5))

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

    function itr4(w)
        w1,w2 = w
        return sum(w1)+sum(w2)
    end
    
    @test gradcheck(itr1,warray)
    @test gradcheck(itr2,warray)
    @test gradcheck(itr1,warray[1])
    @test gradcheck(itr2,warray[1])
    @test gradcheck(itr3,wdict)
    @test gradcheck(itr4,warray) 
    @test gradcheck(itr4,wtuple)
end
