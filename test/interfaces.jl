include("header.jl")

@testset "interfaces" begin

    @testset "Array" begin
        a = rand(3,3)
        for i in (:, 1, 1:2, 1:2:3, [1,2], [2,2], [])
            # @show i
            @test gradcheck(getindex, a, i)
            for j in (:, 1, 1:2, 1:2:3, [1,2], [2,2], [])
                # @show i,j
                @test gradcheck(getindex, a, i, j)
            end
        end
        j = [true,false,true]
        for i in (:, 1, 1:2, 1:2:3, [1,2], [2,2], [])
            # @show i,j
            @test gradcheck(getindex, a, i, j)
            # @show j,i
            @test gradcheck(getindex, a, j, i)
        end
        @test gradcheck(getindex, a, [1 2;1 2])
        @test gradcheck(getindex, a, a.>0.5)
        @test gradcheck(getindex, a, CartesianIndex(1,2))
    end

    @testset "Tuple" begin
        a = Any[rand(3)...]
        t = tuple(a...)
        f(a,i)=(ai=a[i];s=0;for j=1:length(ai); s+=j*ai[j]; end; s)
        g = grad(f)
        for i in (1,1:2,1:2:3,[1,2],[2,2],[],[true,false,true])
            # @show i
            # @test gradcheck(getindex, t, i) # TODO: gradcheck with tuples broken so we compare array vs tuple
            @test g(t,i)==g(a,i)==nothing || g(t,i) == tuple(g(a,i)...)
        end
    end

    @testset "Dict" begin
        g = grad(getindex)
        d = Dict(1=>rand(), 2=>rand(), 3=> rand())
        # @test gradcheck(getindex, d, 1) # TODO: gradcheck with dict broken
        @test collect(g(d,2)) == Any[(2=>1.0)]
    end

    @testset "size" begin
        f0(x) = (p=size(x); p[1]*sum(x.^2))
        @test grad(f0)(ones(3)) == fill(6, 3)

        f1(x)=(p=size(x, 1); p*sum(x.^2))
        @test grad(f1)(ones(3, 3)) == fill(6, 3, 3)

        # issue #18
        f2(x)=(p=size(x, 1, 2); p[1]*sum(x.^2))
        @test grad(f2)(ones(3, 3)) == fill(6, 3, 3)
    end

    @testset "1arg_type" begin
        @test ndims(Rec{Vector{Int}}) == ndims(Vector{Int})
        @test eltype(Rec{Vector{Int}}) == eltype(Vector{Int})
        @test one(Rec{Int}) === one(Int)
        @test zero(Rec{Int}) === zero(Int)
    end
end

nothing
