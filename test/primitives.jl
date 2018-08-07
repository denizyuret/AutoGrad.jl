include("gradcheck.jl")

@testset "primitives" begin
    for t in AutoGrad.alltests()
        @show t
        @test gradcheck(Core.eval(AutoGrad,t[1]), t[2:end]...)
    end
end

nothing
