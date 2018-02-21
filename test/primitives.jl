include("header.jl")

@testset "primitives" begin
    for t in AutoGrad.alltests()
        #@show t
        @test gradcheck(eval(AutoGrad,t[1]), t[2:end]...)
    end

    for A  in (rand(2,2), rand(2,3), rand(3,2))
        @test gradcheck(x->svd(x)[1], A)
        @test gradcheck(x->svd(x)[2], A)
        @test gradcheck(x->svd(x)[3], A)
    end
end

nothing
