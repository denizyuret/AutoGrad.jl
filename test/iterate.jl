include("header.jl")

@testset "iterate" begin
    o = (:delta => 0.01,)

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
    
    for T in (Float32, Float64)
        warray  = [randn(T,2,3,5),randn(T,1,3,5)]
        wtuple  = (randn(T,2,3,5),randn(T,1,3,5))
        wdict   = Dict(:w1=>randn(T,2,3,5),:w2=>randn(T,1,3,5))

        @test gradcheck(itr1,warray; o...)
        @test gradcheck(itr2,warray; o...)
        @test gradcheck(itr1,warray[1]; o...)
        @test gradcheck(itr2,warray[1]; o...)
        @test gradcheck(itr3,wdict; o...)
        @test gradcheck(itr4,warray; o...) 
        @test gradcheck(itr4,wtuple; o...)
    end
end
