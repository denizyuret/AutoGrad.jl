include("header.jl")
using Statistics, LinearAlgebra

@testset "base" begin
    #TO-DO: backslash
    for ft in [Float64,Float32]
        @testset "$(ft)" begin
            atype   = Array{ft}
            x1d     = ft.((1.,2.))
            x2d     = atype.(([1. 2.],[3. 4.]))
            x3d     = (randn(ft,2,3,5),randn(ft,1,3,5))
            x4d     = (randn(ft,2,3,5,4),randn(ft,2,3,1,4))
            xsquare = rand(ft,3,3)

            bt(x...) = broadcast(*,x...)
            ba(x...) = broadcast(+,x...)
            bm(x...) = broadcast(-,x...)
            bd(x...) = broadcast(/,x...)
            bp(x...) = broadcast(^,x...)

            @testset "product" begin
                @test gradcheckN(*,x1d...)
                @test gradcheckN(*,x1d[1],x2d[2])
                @test gradcheckN(bt,x1d[1],x2d[2])
                @test gradcheckN(*,x2d[1],x2d[2]')
                @test gradcheckN(bt,x2d...)
                @test gradcheckN(bt,x3d...)
                @test gradcheckN(bt,x4d...)
            end

            @testset "division" begin
                @test gradcheckN(/,x1d...)
                @test gradcheckN(bd,x1d[1],x2d[2])
                @test gradcheckN(bd,x2d[1]...)
                @test gradcheckN(bd,x2d...)
                @test gradcheckN(bd,x3d...)
                @test gradcheckN(bd,x4d...) # fails for Float32
            end

            @testset "plus/minus" begin
                for op in [+,-]
                    @test gradcheckN(op,x1d[1],x1d[2])
                    @test gradcheckN(op,x2d...)
                    @test gradcheckN(op,x3d[1],copy(x3d[1]))
                    @test gradcheckN(op,x4d[1],copy(x4d[1]))
                end
                for op in [ba,bm]
                    @test gradcheckN(op,x1d[1],x2d[2])
                    @test gradcheckN(op,x2d...)
                    @test gradcheckN(op,x3d...)
                    @test gradcheckN(op,x4d...)
                end
            end

            @testset "power" begin
                @test_broken gradcheckN(^,xsquare,x1d[2]) #TODO: integer and matrix powers
                @test gradcheckN(bp,abs.(x2d[1]),x1d[2])
                @test gradcheckN(bp,abs.(x3d[1]),x1d[2])
                @test gradcheckN(bp,abs.(x4d[1]),x1d[2])
            end

            @testset "values" begin
                d = Dict(:a=>1, :b=>2)
                @test grad(x->sum(values(x)))(d) == Dict(:a=>1, :b=>1)
            end

            @test gradcheck(abs,x1d[1])
            @test gradcheck(abs2,x1d[1])
            @test gradcheck(big,x1d[1])
            @test gradcheck(float,1)
            @test gradcheck(maximum,x4d[1])
            @test gradcheck(x->maximum(abs,x),x4d[1])
            @test gradcheck(minimum,x4d[1])
            @test gradcheck(x->minimum(abs,x),x4d[1]) #fails for Float32
            @test gradcheck(permutedims,x4d[1],(3,4,2,1))
            @test_broken gradcheck(prod,x1d) #TODO: gradcheck tuple support so collect not necessary in next line
            @test gradcheck(prod,collect(x1d))
            @test gradcheck(sum,x4d[1])
            @test gradcheck(x->sum(abs,x),x4d[1])
            @test gradcheck(x->sum(abs2,x),x4d[1])
            @test gradcheck(vec,x4d[1])
            @test gradcheck(copy,x4d[1])
        end
    end
end
