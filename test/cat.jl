include("header.jl")

@testset "cat" begin
    cat1(x...)=cat(x...; dims=Val(1))
    cat2(x...)=cat(x...; dims=Val(2))
    cat33(x...)=cat(x...; dims=Val(3))

    for ft in [Float32,Float64]
        @testset "$(ft)" begin

            atype = Array{ft}
            x1d = ft.((1.,2.))
            x2d = atype.(([1. 2.],[3. 4.]))
            x3d = atype.((randn(2,3,5),randn(1,3,5),randn(2,2,5),randn(2,3,2)))
            xv  = atype.(([1.,2.],[3.,4.]))


            @test gradcheckN(cat1, x1d...)
            @test gradcheckN(cat1, x1d[1], x2d[1]')
            @test_skip gradcheckN(cat1, x1d[1], x2d[1]) #TODO: !!uncat mismatch error!! why is this working in Base.cat?

            @test gradcheckN(cat2, x1d...)
            @test gradcheckN(cat2, x1d[1], x2d[1])
            @test gradcheckN(cat2, x2d...)

            @test gradcheckN(cat1, x3d[1],x3d[2])
            @test gradcheckN(cat2, x3d[1],x3d[3])
            @test gradcheckN(cat33, x3d[1],x3d[4])

            @test gradcheckN(cat1d,xv...)
        end
    end
end
