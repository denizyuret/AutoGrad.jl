include("header.jl")
using LinearAlgebra

@testset "LinearAlgebra" begin

    w = randn(3,2)
    wt = randn(2,3)
    wsquare = randn(3,3)
    udg = [1.,2.,3.]
    dg = [4.,5.,6.]
    ldg  = [7.,8.,9.]

    diagm1(x,y,z) = diagm(1=>x,0=>y,-1=>z)

    function krontest(w1,w2)
        sum(kron(w1,w2))
    end

    function qrtest(w) 
        Q,R = qr(w)
        sum(Q) + 2sum(R)
    end

    function lqtest(w) 
        Q,L = lq(w)
        sum(Q) + 2sum(L)
    end

    function svdtest(w) 
        U,S,V = svd(w)
        sum(U) + 2sum(S) + 3sum(V)
    end

    @test gradcheck(adjoint,w)
    @test gradcheck(det,wsquare)
    #@test gradcheck(diag,w)
    #@test gradcheckN(diagm1,udg,dg,ldg)
    @test gradcheckN(dot,udg,ldg)
    @test gradcheckN(dot,w,copy(w))
    @test gradcheck(inv,wsquare)
    #@test gradcheck(krontest,w,copy(w))
    @test gradcheck(logabsdet,wsquare)
    @test gradcheck(logdet,wsquare)
    @test gradcheck(norm,w,1)
    @test gradcheck(norm,w,2)
    @test gradcheck(norm,w,Inf)
   #@test gradcheck(qrtest,w) #iterator error
   #@test gradcheck(lqtest,w) #iterator error
   #@test gradcheck(svdtest,w) #iterator error
    @test gradcheck(tr,wsquare)
    @test gradcheck(transpose,w)
    @test gradcheck(tril,w)
    @test gradcheck(triu,w)
    @test gradcheckN(*,w,wt)
end
