include("header.jl")

@testset "primitives" begin
for t in AutoGrad.alltests()
        @test gradcheck(eval(AutoGrad,t.f), t.args...; kwargs = t.kargs)
    end
end

nothing
