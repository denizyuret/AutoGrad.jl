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

    @testset "product" begin
        @test gradcheck(*,x1d...)
        @test gradcheck(*,x1d[1],x2d[2])
        @test gradcheck(bt,x1d[1],x2d[2])
        @test gradcheck(*,x2d[1],x2d[2]')
        @test gradcheck(bt,x2d...)
        @test gradcheck(bt,x3d...)
        @test gradcheck(bt,x4d...)
    end

    @testset "division" begin
        @test gradcheck(/,x1d...)
        @test gradcheck(bd,x1d[1],x2d[2])
        @test gradcheck(bd,x2d[1]...)
        @test gradcheck(bd,x2d...)
        @test gradcheck(bd,x3d...)
        @test gradcheck(bd,x4d...)
    end

    @testset "plus/minus" begin
        for op in [+,-]
            @test gradcheck(op,x1d[1],x1d[2])
            @test gradcheck(op,x2d...)
            @test gradcheck(op,x3d[1],copy(x3d[1]))
            @test gradcheck(op,x4d[1],copy(x4d[1]))
        end
        for op in [ba,bm]
            @test gradcheck(op,x1d[1],x2d[2])
            @test gradcheck(op,x2d...)
            @test gradcheck(op,x3d...)
            @test gradcheck(op,x4d...)
        end
    end

    @testset "power" begin
        @test_broken gradcheck(^,xsquare,x1d[2]) #TODO: integer and matrix powers
        @test gradcheck(bp,abs.(x2d[1]),x1d[2])
        @test gradcheck(bp,abs.(x3d[1]),x1d[2])
        @test gradcheck(bp,abs.(x4d[1]),x1d[2])
    end

    @testset "values" begin
        d = Dict(:a=>1, :b=>2)
        @test grad(x->sum(values(x)))(d) == Dict(:a=>1, :b=>1)
    end

    @test gradcheck(abs,x1d[1])
    @test gradcheck(abs2,x1d[1])
    @test gradcheck(big,x1d[1])
    @test gradcheck(float,1)
    @test gradcheck(maximum,x4d[1])
    @test gradcheck(x->maximum(abs,x),x4d[1])
    @test gradcheck(minimum,x4d[1])
    @test gradcheck(x->minimum(abs,x),x4d[1])
    @test gradcheck(permutedims,x4d[1],(3,4,2,1); args=1)
    @test gradcheck(prod,x1d)
    @test gradcheck(sum,x4d[1])
    @test gradcheck(x->sum(abs,x),x4d[1])
    @test gradcheck(x->sum(abs2,x),x4d[1])
    @test gradcheck(vec,x4d[1])
    @test gradcheck(copy,x4d[1])
end
