include("header.jl")
using Statistics, LinearAlgebra

@testset "base" begin
    #TO-DO: backlash
    x1d = (1.,2.)
    x2d = ([1. 2.],[3. 4.]) 
    x3d = (randn(2,3,5),randn(1,3,5)) 
    x4d = (randn(2,3,5,4),randn(2,3,1,4))
    xsquare = rand(3,3)

    bt(x...) = broadcast(*,x...) 
    ba(x...) = broadcast(+,x...) 
    bm(x...) = broadcast(-,x...) 
    bd(x...) = broadcast(/,x...)
    bp(x...) = broadcast(^,x...)


    minabs(x) = minimum(abs,x)
    maxabs(x) = maximum(abs,x)
    sumabs(x) = sum(abs,x) 
    sumabs2(x) = sum(abs2,x) 

    @testset "product" begin
        @test gradcheckN(*,x1d...)
        @test gradcheckN(*,x1d[1],x2d[2])
        @test gradcheckN(bt,x1d[1],x2d[2])
        @test gradcheckN(*,x2d[1],x2d[2]')
        @test gradcheckN(bt,x2d...)
        @test gradcheckN(bt,x3d...)
        @test gradcheckN(bt,x4d...)
    end

    @testset "division" begin
        @test gradcheckN(/,x1d...)
        @test gradcheckN(bd,x1d[1],x2d[2])
        @test gradcheckN(bd,x2d[1]...)
        @test gradcheckN(bd,x2d...)
        @test gradcheckN(bd,x3d...)
        @test gradcheckN(bd,x4d...)
    end

    @testset "plus/minus" begin
        for op in [+,-]
            @test gradcheckN(op,x1d[1],x1d[2])
            @test gradcheckN(op,x2d...)
            @test gradcheckN(op,x3d[1],copy(x3d[1]))
            @test gradcheckN(op,x4d[1],copy(x4d[1]))
        end
        for op in [ba,bm]
            @test gradcheckN(op,x1d[1],x2d[2])
            @test gradcheckN(op,x2d...)
            @test gradcheckN(op,x3d...)
            @test gradcheckN(op,x4d...)
        end
    end

    @testset "power" begin
        @test gradcheckN(^,xsquare,x1d[2]) #fails
        @test gradcheckN(bp,abs.(x2d[1]),x1d[2])
        @test gradcheckN(bp,abs.(x3d[1]),x1d[2])
        @test gradcheckN(bp,abs.(x4d[1]),x1d[2])
    end
    @test gradcheck(abs,x1d[1])
    @test gradcheck(abs2,x1d[1])
    @test gradcheck(big,x1d[1]) #fails
    @test gradcheck(float,1) #fails
    @test gradcheck(maximum,x4d[1])
    @test gradcheck(maxabs,x4d[1])
    @test gradcheck(minimum,x4d[1])
    @test gradcheck(minabs,x4d[1])
    @test gradcheck(permutedims,x4d[1],(3,4,2,1))
    @test gradcheckN(prod,x1d) #fails
    @test gradcheck(sum,x4d[1])
    @test gradcheck(sumabs,x4d[1])
    @test gradcheck(sumabs2,x4d[1])
    @test gradcheck(vec,x4d[1])
    @test gradcheck(copy,x4d[1])
end
