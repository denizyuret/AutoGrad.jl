include("header.jl")

@testset "getindex" begin

    for T in (Float32, Float64)
        @testset "Array" begin
            a = rand(T,3,3)
            for i in (:, 1, 1:2, 1:2:3, [1,2], [2,2], [])
                #@show i
                @test gradcheck(getindex, a, i; args=1)
                for j in (:, 1, 1:2, 1:2:3, [1,2], [2,2], [])
                    #@show i,j
                    @test gradcheck(getindex, a, i, j; args=1)
                end
            end
            j = [true,false,true]
            for i in (:, 1, 1:2, 1:2:3, [1,2], [2,2], [])
                #@show i,j
                @test gradcheck(getindex, a, i, j; args=1)
                #@show j,i
                @test gradcheck(getindex, a, j, i; args=1)
            end
            @test gradcheck(getindex, a, [1 2;1 2]; args=1)
            @test gradcheck(getindex, a, a.>0.5; args=1)
            @test gradcheck(getindex, a, CartesianIndex(1,2); args=1)
        end

        @testset "Tuple" begin
            a = Any[rand(T,3)...]
            t = tuple(a...)
            f(a,i)=(ai=a[i];s=0;for j=1:length(ai); s+=j*ai[j]; end; s)
            g = grad(f)
            for i in (1,1:2,1:2:3,[1,2],[2,2],[],[true,false,true])
                #@show i
                @test gradcheck(getindex, t, i; args=1)
                @test g(t,i)==g(a,i)==nothing || g(t,i) == tuple(g(a,i)...)
            end
        end

        @testset "Dict" begin
            g = grad(getindex)
            d = Dict(1=>rand(T), 2=>rand(T), 3=> rand(T))
            #@show d
            @test gradcheck(getindex, d, 1; args=1)
            @test collect(g(d,2)) == Any[(2=>1.0)]
        end

        @testset "size" begin
            #@show size
            f0(x)=(p=size(x); p[1]*sum(abs2,x))
            @test grad(f0)(ones(3)) == fill(6, 3)

            f1(x)=(p=size(x, 1); p*sum(abs2,x))
            @test grad(f1)(ones(3, 3)) == fill(6, 3, 3)

            # Issue #18: ambiguity error in size(rec, dims...)
            f2(x)=(p=(size(x, 1), size(x, 2)); p[1]*sum(abs2,x))
            @test grad(f2)(ones(3, 3)) == fill(6, 3, 3)
        end

        @testset "indexing" begin

            # info("Test indexing...")

            a1 = rand(T,3)
            t1 = (a1...,)
            d1 = Dict(); for i=1:length(a1); d1[i]=a1[i]; end

            s0(x)=x[1]^2+x[2]^2
            s1 = grad(s0)
            s1sum(x)=(y=s1(x);y[1]+y[2])
            s2 = grad(s1sum)

            @test gradcheck(s0,a1)
            @test gradcheck(s0,t1)
            @test gradcheck(s0,d1)
            @test gradcheck(s1sum,a1)
            @test gradcheck(s1sum,t1)
            @test gradcheck(s1sum,d1)

            f0(x)=(a=0;for i=1:length(x);a+=x[i]^2;end;a)
            f1=grad(f0)
            f1sum(x)=sum(values(f1(x)))
            f2=grad(f1sum)
            @test gradcheck(f0,a1)
            @test gradcheck(f0,t1)
            @test gradcheck(f0,d1)
            @test gradcheck(f1sum,a1)
            @test gradcheck(f1sum,t1)
            @test gradcheck(f1sum,d1)

            r0(x)=(s=0; for i=2:length(x); s+=(1-x[i-1])^2 + 2*(x[i]-x[i-1]^2)^2; end; s)
            r1 = grad(r0)
            r1sum(x)=sum(values(r1(x)))
            r2 = grad(r1sum)
            @test gradcheck(r0,a1)
            @test gradcheck(r0,t1)
            @test gradcheck(r0,d1)
            @test gradcheck(r1sum,a1)
            @test gradcheck(r1sum,t1)
            @test gradcheck(r1sum,d1)

        end

        @testset "view" begin
            a = rand(T,3,3)    
            g1 = grad(x->sum(x[1:2,1:2].^2.5))
            g2 = grad(x->sum(view(x,1:2,1:2).^2.5))
            @test g1(a) == g2(a)

            g1 = grad(x->sum(x[:,1:2].^2.0))
            g2 = grad(x->sum(view(x,:,1:2).^2.0))
            g3 = grad(x->sum(selectdim(x,2,1:2).^2.0))
            @test g1(a) == g2(a) == g3(a)
        end
    end

    # Issue #73: incorrect gradient when indexing into a matrix of vectors
    a = Matrix{Array{Float32}}(undef, 2, 1)
    a[1] = [2,3,4]
    a[2] = [4,5,6]
    g1 = grad(x->x[1]' * x[2])
    g2 = grad(x->x[1, 1]' * x[2, 1])
    @test g1(a) == g2(a)

end

nothing
