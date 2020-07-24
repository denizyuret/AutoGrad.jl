include("header.jl")

@testset "sparse" begin

    # Issue #114 (a): plus for Sparse.
    using AutoGrad: Sparse, full, addto!
    a = Sparse(zeros(3,4), Any[ [1.,1.], [1.,1.], 1., 1. ], Any[ ([1,2],), (3:4,), (2,2), (1,) ])
    b = a + a
    @test b isa Sparse
    @test full(b) == full(a) + full(a)
    addto!(a, a)
    @test a isa Sparse
    @test full(a) == full(b)
    b = a + a
    a .+= a
    @test full(a) == full(b)

    # Issue #114 (b): lmul! ambiguous for Sparse, breaks gclip.
    using LinearAlgebra
    foo(w) = (s = 0.0; for i=1:length(w); s+=w[i]; end; s)
    w = Param(randn(2,2))
    J = @diff foo(w)
    @test lmul!(0.5, grad(J,w)) isa Sparse

    # Issue #116: Sparse values and indices should be Vector{Any}
    foo(w) = sum(w[:,[1,2]]) + sum(w[:,[1 2]])
    w = Param(randn(2,2))
    @test @gcheck foo(w)

    # Issue #121: Sparse values and indices should be Vector{Any}
    w,x,y = Param(randn(7)), randn(10), randn(10)
    pred(w,x)=w[5:6]'*tanh.(w[1:2]*x' .+ w[3:4]) .+ w[7]
    @test @gcheck mean(abs2,y-pred(w,x)[1,:])
end
