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
end
