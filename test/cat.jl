include("header.jl")

@testset "cat" begin
    cat1(x...)=cat(x...; dims=Val(1))
    cat2(x...)=cat(x...; dims=Val(2))
    cat31(x...)=cat(x...; dims=Val(1))
    cat32(x...)=cat(x...; dims=Val(2))
    cat33(x...)=cat(x...; dims=Val(3))

    for T in (Float32, Float64)
        x1d = (randn(T), randn(T))
        x2d = (randn(T,2), randn(T,2))
        x3d = (randn(T,2,3,5),randn(T,1,3,5),randn(T,2,2,5),randn(T,2,3,2)) 

        @test gradcheck(cat1, x1d...)
        @test gradcheck(cat1, x2d...)
        @test gradcheck(cat1, x1d[1], x2d[1])
        
        @test gradcheck(cat2, x1d...)
        @test gradcheck(cat2, x2d...)
        @test gradcheck(cat2, x1d[1], x2d[1]')
        
        @test gradcheck(cat31, x3d[1],x3d[2])
        @test gradcheck(cat32, x3d[1],x3d[3])
        @test gradcheck(cat33, x3d[1],x3d[4])
        
        @test gradcheck(cat1d,T[1.,2.],T[3.,4.])
    end
end

