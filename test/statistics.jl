include("header.jl")

using Statistics
meanabs(x)=mean(abs,x)
meanabs2(x)=mean(abs2,x)

@testset "statistics" begin
    @test gradcheck(mean, randn(2,3))
    @test gradcheck(mean, randn(2,3), kwargs=[:dims=>1])
    @test gradcheck(mean, randn(2,3), kwargs=[:dims=>(1,2)])
    @test gradcheck(meanabs, randn(2,3))
    @test gradcheck(meanabs2, randn(2,3))
    @test gradcheck(var, randn(2,3))
    @test gradcheck(var, randn(2,3), kwargs=[:dims=>1])
    @test gradcheck(var, randn(2,3), kwargs=[:dims=>(1,2)])
    @test gradcheck(std, randn(2,3))
    @test gradcheck(std, randn(2,3), kwargs=[:dims=>1])
    @test gradcheck(std, randn(2,3), kwargs=[:dims=>(1,2)])
end
