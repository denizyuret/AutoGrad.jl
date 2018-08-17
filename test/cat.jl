include("header.jl")
cat1(x...)=cat(x...; dims=Val(1))
cat2(x...)=cat(x...; dims=Val(2))
cat31(x...)=cat(x...; dims=Val(1))
cat32(x...)=cat(x...; dims=Val(2))
cat33(x...)=cat(x...; dims=Val(3))

x1d = (1.,2.)
x2d = ([1. 2.],[3. 4.]) 
x3d = (randn(2,3,5),randn(1,3,5),randn(2,2,5),randn(2,3,2)) 

@testset "cat" begin
    @test gradcheckN(cat1, x1d...)
    #@test gradcheckN(cat1, x1d[1], x2d[1]) #uncat mismatch error
    
    @test gradcheckN(cat2, x1d...)
    @test gradcheckN(cat2, x1d[1], x2d[1])
    @test gradcheckN(cat2, x2d...)
    
    @test gradcheckN(cat31, x3d[1],x3d[2])
    @test gradcheckN(cat32, x3d[1],x3d[3])
    @test gradcheckN(cat33, x3d[1],x3d[4])
    
    @test gradcheckN(cat1d,[1.,2.],[3.,4.])
end

