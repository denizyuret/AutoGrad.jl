include("header.jl")

@testset "sparse" begin
    # Issue 114
    using AutoGrad: Sparse, full, addto!
    a = Sparse(zeros(3,4), [ [1.,1.], [1.,1.], 1., 1. ], [ ([1,2],), (3:4,), (2,2), (1,) ])
    b = a + a
    @test b isa Sparse
    @test full(b) == full(a) + full(a)
    addto!(a, a)
    @test a isa Sparse
    @test full(a) == full(b)
    b = a + a
    a .+= a
    @test full(a) == full(b)
end
