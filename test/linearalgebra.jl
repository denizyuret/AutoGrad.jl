include("header.jl")
using LinearAlgebra

@testset "LinearAlgebra" begin

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

    for T in (Float32, Float64)
        w = randn(T,3,2)
        wt = randn(T,2,3)
        wsquare = I + 0.2*randn(T,3,3)
        wposdef = wsquare' * wsquare
        udg = T[1.,2.,3.]
        dg = T[4.,5.,6.]
        ldg  = T[7.,8.,9.]

        @test gradcheck(adjoint,w)
        @test gradcheck(det,wsquare)
        @test gradcheck(diag,wsquare)
        @test_skip gradcheck(diag,w)             #TODO no support for non-square matrices yet
        @test_skip gradcheck(diagm1,udg,dg,ldg)   #TODO diagm not implemented yet
        @test gradcheck(dot,udg,ldg)
        @test gradcheck(dot,w,copy(w))
        @test gradcheck(inv,wsquare)
        @test gradcheck(krontest,w,copy(w); delta=0.01)
        @test gradcheck(logabsdet,wsquare)
        @test gradcheck(logdet,wposdef)
        @test gradcheck(norm,w,1; args=1)
        @test gradcheck(norm,w,2; args=1)
        @test gradcheck(norm,w,Inf; args=1)
        @test_skip gradcheck(qrtest,w) #TODO iterator error
        @test_skip gradcheck(lqtest,w) #TODO iterator error
        @test_skip gradcheck(svdtest,w) #TODO iterator error
        @test gradcheck(tr,wsquare)
        @test gradcheck(transpose,w)
        @test gradcheck(tril,w)
        @test gradcheck(triu,w)
        @test gradcheck(*,w,wt)
        @test_skip gradcheck(^,wsquare,randn()) #TODO: real matrix powers
        @test_skip gradcheck(^,wsquare,rand(-3:3)) #TODO: integer matrix powers
    end
end
