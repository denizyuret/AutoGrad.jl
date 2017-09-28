include("header.jl")

@testset "primitives" begin
    for t in AutoGrad.alltests()
        #@show t
        @test gradcheck(eval(AutoGrad,t[1]), t[2:end]...)
    end
end

nothing
