include("header.jl")

using Statistics
meanabs(x)=mean(abs,x)
meanabs2(x)=mean(abs2,x)

@testset "statistics" begin
    o = (:delta=>0.0001,:atol=>0.01,:rtol=>0.01)
    for T in (Float32,Float64)
        @test gradcheck(mean, randn(T,2,3); o...)
        @test gradcheck(mean, randn(T,2,3); kw=[:dims=>1], o...)
        @test gradcheck(mean, randn(T,2,3); kw=[:dims=>(1,2)], o...)
        @test gradcheck(meanabs, randn(T,2,3); o...)
        @test gradcheck(meanabs2, randn(T,2,3); o...)
        @test gradcheck(var, randn(T,2,3); o...)
        @test gradcheck(var, randn(T,2,3); kw=[:dims=>1], o...)
        @test gradcheck(var, randn(T,2,3); kw=[:dims=>(1,2)], o...)
        @test gradcheck(std, randn(T,2,3); o...)
        @test gradcheck(std, randn(T,2,3); kw=[:dims=>1], o...)
        @test gradcheck(std, randn(T,2,3); kw=[:dims=>(1,2)], o...)
    end
end
