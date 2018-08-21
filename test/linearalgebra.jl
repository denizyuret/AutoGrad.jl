include("header.jl")
using LinearAlgebra

@testset "LinearAlgebra" begin
    for ft in [Float64,Float32]
        @testset "$(ft)" begin
            w       = randn(ft,3,2)
            wt      = randn(ft,2,3)
            wsquare = randn(ft,3,3)
            wposdef = wsquare' * wsquare
            udg     = ft.([1.,2.,3.])
            dg      = ft.([4.,5.,6.])
            ldg     = ft.([7.,8.,9.])

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
            @test gradcheck(diag,wsquare)
            @test_broken gradcheck(diag,w)             #TODO no support for non-square matrices yet
            @test_skip gradcheckN(diagm1,udg,dg,ldg)   #TODO diagm not implemented yet
            @test gradcheckN(dot,udg,ldg)
            @test gradcheckN(dot,w,copy(w))
            @test gradcheck(inv,wsquare)
            @test gradcheck(krontest,w,copy(w))
            @test gradcheck(logabsdet,wsquare)
            @test gradcheck(logdet,wposdef) #fails on Float32
            @test gradcheck(norm,w,1)
            @test gradcheck(norm,w,2)
            @test gradcheck(norm,w,Inf)
            @test_broken gradcheck(qrtest,w) #TODO iterator error
            @test_broken gradcheck(lqtest,w) #TODO iterator error
            @test_broken gradcheck(svdtest,w) #TODO iterator error
            @test gradcheck(tr,wsquare)
            @test gradcheck(transpose,w)
            @test gradcheck(tril,w)
            @test gradcheck(triu,w)
            @test gradcheckN(*,w,wt)
        end
    end
end
