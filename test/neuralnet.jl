include("header.jl")

@testset "neuralnet" begin
    # info("Test neural net...")
    n0(w,x,y)=sum(abs2,((w[3]*max.(0,w[1]*x.+w[2]).+w[4])-y))
    n1 = grad(n0)
    n1sum(w,x,y)=sum(map(sum,n1(w,x,y)))
    n1sumd(w,x,y)=sum(map(sum,values(n1(w,x,y))))
    for ft in [Float64,Float32]
        @testset "$(ft)" begin
            wa = Any[rand(ft,2,3),rand(ft,2),rand(ft,2,2),rand(ft,2)]
            wt = (wa...,)
            wd = Dict(); for i=1:length(wa); wd[i]=wa[i]; end

            @test gradcheckN(n0, wa, rand(ft,3,10), rand(ft,2,10))
            @test gradcheckN(n0, wt, rand(ft,3,10), rand(ft,2,10))
            @test gradcheckN(n0, wd, rand(ft,3,10), rand(ft,2,10))
            @test gradcheckN(n1sum, wa, rand(ft,3,10), rand(ft,2,10))
            @test gradcheckN(n1sum, wt, rand(ft,3,10), rand(ft,2,10)) #fails at Float32
            # TODO: This needs more work:
            @test_skip check_grads(n1sumd, wd, rand(ft,3,10), rand(ft,2,10))  # TODO: FAIL
        end
    end
end
nothing
