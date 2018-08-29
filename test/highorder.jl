include("header.jl")
@testset "highorder" begin
    # info("Test higher order gradients...")
    g1 = grad(sin); @test g1(1)==cos(1)
    g2 = grad(g1);  @test g2(1)==-sin(1)
    g3 = grad(g2);  @test g3(1)==-cos(1)
    g4 = grad(g3);  @test g4(1)==sin(1)
    g5 = grad(g4);  @test g5(1)==cos(1)
    g6 = grad(g5);  @test g6(1)==-sin(1)
    g7 = grad(g6);  @test g7(1)==-cos(1)
    g8 = grad(g7);  @test g8(1)==sin(1)
    g9 = grad(g8);  @test g9(1)==cos(1)

    # PR #75: Tape confusion fix
    @test grad(x -> x*grad(y -> x+y)(x))(5.0) == 1
    @test grad(x -> x*grad(y -> x+y)(1x))(5.0) == 1
    
    # WIP
    # @test (s->gradient((@diff (x->x*(t->gradient((@diff (y->x+y)(t)),t))(x))(s)),s))(Param(5)) == 1
    # @test (s->gradient((@diff (x->x*(t->gradient((@diff (y->x+y)(t)),t))(1x))(s)),s))(Param(5)) == 1
    # @test (s->gradient(differentiate(x->x*(t->gradient(differentiate(y->x+y,t),t))(x),s),s))(Param(5)) == 1
    # @test (s->gradient(differentiate(x->x*(t->gradient(differentiate(y->x+y,t),t))(1x),s),s))(Param(5)) == 1

    # Issue #44: third gradient of exp(x*x) gives nothing
    @test exp(1) == grad(exp)(1) == grad(grad(exp))(1)
    f(x) = exp(x*x)
    @test f(1) ≈ grad(f)(1) / 2 ≈ grad(grad(f))(1) / 6

    # Issue #62: bug second derivative tanh
    @test grad(tanh)(1) == 1 - tanh(1)^2
    @test grad(grad(tanh))(1) == -2*tanh(1)*(1-tanh(1)^2)
end

nothing
