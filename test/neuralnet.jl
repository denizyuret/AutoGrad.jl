include("header.jl")

# info("Test neural net...")
n0(w,x,y)=sum(((w[3]*max.(0,w[1]*x.+w[2]).+w[4])-y).^2)
n1 = grad(n0)
n1sum(w,x,y)=sum(map(sum,n1(w,x,y)))
n1sumd(w,x,y)=sum(map(sum,values(n1(w,x,y))))
wa = Any[rand(2,3),rand(2),rand(2,2),rand(2)]
wt = (wa...,)
wd = Dict(); for i=1:length(wa); wd[i]=wa[i]; end

@testset "neuralnet" begin
    @test gradcheckN(n0, wa, rand(3,10), rand(2,10))
    @test gradcheckN(n0, wt, rand(3,10), rand(2,10))
    @test gradcheckN(n0, wd, rand(3,10), rand(2,10))
    @test gradcheckN(n1sum, wa, rand(3,10), rand(2,10))
    @test gradcheckN(n1sum, wt, rand(3,10), rand(2,10))
    # TODO: This needs more work:
    # @test check_grads(n1sumd, wd, rand(3,10), rand(2,10))  # TODO: FAIL
end

nothing
