include("header.jl")
using Statistics, LinearAlgebra
#TO-DO: backslash

@testset "base" begin

    # to prevent overflow with division
    function randn0(d...; ϵ=0.5)
        x = randn(d...)
        x = sign.(x) .* (abs.(x) .+ ϵ)
        return x
    end

    bt(x...) = broadcast(*,x...) 
    ba(x...) = broadcast(+,x...) 
    bm(x...) = broadcast(-,x...) 
    bd(x...) = broadcast(/,x...)
    bp(x...) = broadcast(^,x...)

    for T in (Float32, Float64)
        x1d = randn0(T,2)
        x2d = (randn0(T,2), randn0(T,2))
        x3d = (randn0(T,2,3,5),randn0(T,1,3,5)) 
        x4d = (randn0(T,2,3,5,4),randn0(T,2,3,1,4))
        xsquare = randn0(T,3,3)

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
            @test gradcheck(^, abs(rand()), randn())
            @test gradcheck(^, abs(rand()), rand(-3:3))
            @test gradcheck(bp,abs.(x2d[1]),x1d[2])
            @test gradcheck(bp,abs.(x3d[1]),x1d[2])
            @test gradcheck(bp,abs.(x4d[1]),x1d[2])
            # Move these to linearalgebra.jl:
            # @test_broken gradcheck(^,xsquare,randn())
            # @test_broken gradcheck(^,xsquare,rand(-3:3))
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

    # Issue #76: StackOverflowError in grad on broadcasted function
    f(x) = x.^2
    jf = AutoGrad.grad(f)
    @test jf(1) == 2

    # Issue #80: broadcast error for integer power
    @test grad(x->sum(x.^2))([1,2,3]) == [2,4,6]
    @test grad(x->sum(x.^2.0))([1,2,3]) == [2.,4.,6.]
    r = [1. 2.; 3. 4.]
    @test_throws ErrorException grad(x->sum(x^2.0))(r)
    @test_throws ErrorException grad(x->sum(x^2))(r)
    @test grad(x->sum(x.^2))(r) == [2.0 4.0; 6.0 8.0]
    @test grad(x->sum(x.^2.0))(r) == [2.0 4.0; 6.0 8.0]
    @test isa(r^3.1, Array{Complex{Float64},2})
    @test isa(r.^3.1, Array{Float64,2})
    @test_throws ErrorException grad(x->sum(x^3.1))(r)
    @test isa(grad(x->sum(x.^3.1))(r), Array{Float64,2})

end
