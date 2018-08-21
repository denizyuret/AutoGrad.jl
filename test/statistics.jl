include("header.jl")

using Statistics

@testset "statistics" begin
    for ft in [Float64,Float32]
        @testset "$(ft)" begin
            @test gradcheck(mean, randn(ft,2,3))
            @test gradcheck(mean, randn(ft,2,3), kwargs=[:dims=>1])
            @test gradcheck(mean, randn(ft,2,3), kwargs=[:dims=>(1,2)])
            @test gradcheck(x->mean(abs,x), randn(ft,2,3))
            @test gradcheck(x->mean(abs2,x), randn(ft,2,3))
            @test gradcheck(var, randn(ft,2,3))
            @test gradcheck(var, randn(ft,2,3), kwargs=[:dims=>1])
            @test gradcheck(var, randn(ft,2,3), kwargs=[:dims=>(1,2)])
            @test gradcheck(std, randn(ft,2,3))
            @test gradcheck(std, randn(ft,2,3), kwargs=[:dims=>1]) # fails on Float32
            @test gradcheck(std, randn(ft,2,3), kwargs=[:dims=>(1,2)])
        end
    end
end
