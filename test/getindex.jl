include("header.jl")

@testset "getindex" begin

    @testset "Array" begin
        a = rand(3,3)
        for i in (:, 1, 1:2, 1:2:3, [1,2], [2,2], [])
            #@show i
            @test gradcheck(getindex, a, i)
            for j in (:, 1, 1:2, 1:2:3, [1,2], [2,2], [])
                #@show i,j
                @test gradcheck(getindex, a, i, j)
            end
        end
        j = [true,false,true]
        for i in (:, 1, 1:2, 1:2:3, [1,2], [2,2], [])
            #@show i,j
            @test gradcheck(getindex, a, i, j)
            #@show j,i
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
            #@show i
            # @test gradcheck(getindex, t, i) # TODO: gradcheck with tuples broken so we compare array vs tuple
            @test g(t,i)==g(a,i)==nothing || g(t,i) == tuple(g(a,i)...)
        end
    end

    @testset "Dict" begin
        g = grad(getindex)
        d = Dict(1=>rand(), 2=>rand(), 3=> rand())
        #@show d
        # @test gradcheck(getindex, d, 1) # TODO: gradcheck with dict broken
        @test collect(g(d,2)) == Any[(2=>1.0)]
    end

    @testset "size" begin
        #@show size
        f0(x)=(p=size(x); p[1]*sum(abs2,x))
        @test grad(f0)(ones(3)) == fill(6, 3)

        f1(x)=(p=size(x, 1); p*sum(abs2,x))
        @test grad(f1)(ones(3, 3)) == fill(6, 3, 3)

        # issue #18
        f2(x)=(p=(size(x, 1), size(x, 2)); p[1]*sum(abs2,x))
        @test grad(f2)(ones(3, 3)) == fill(6, 3, 3)
    end

    #= why do we need this?
    @testset "1arg_type" begin
        @test ndims(Rec{Vector{Int}}) == ndims(Vector{Int})
        @test eltype(Rec{Vector{Int}}) == eltype(Vector{Int})
        @test one(Rec{Int}) === one(Int)
        @test zero(Rec{Int}) === zero(Int)
    end
    =#

    @testset "indexing" begin

        # info("Test indexing...")

        a1 = rand(3)                    # TODO: FAIL: some high-order (b1sum) tests with rand(2)
        t1 = (a1...,)
        d1 = Dict(); for i=1:length(a1); d1[i]=a1[i]; end

        s0(x)=x[1]^2+x[2]^2
        s1 = grad(s0)
        s1sum(x)=(y=s1(x);y[1]+y[2])
        s2 = grad(s1sum)

        @test check_grads(s0,a1)
        @test check_grads(s0,t1)
        @test check_grads(s0,d1)
        @test check_grads(s1sum,a1)
        @test check_grads(s1sum,t1)
        @test check_grads(s1sum,d1)

        f0(x)=(a=0;for i=1:length(x);a+=x[i]^2;end;a)
        f1=grad(f0)
        f1sum(x)=sumvalues(f1(x))
        f2=grad(f1sum)
        @test check_grads(f0,a1)
        @test check_grads(f0,t1)
        @test check_grads(f0,d1)
        @test check_grads(f1sum,a1)
        @test check_grads(f1sum,t1)
        @test check_grads(f1sum,d1)

        r0(x)=(s=0; for i=2:length(x); s+=(1-x[i-1])^2 + 100*(x[i]-x[i-1]^2)^2; end; s)
        r1 = grad(r0)
        r1sum(x)=sumvalues(r1(x))
        r2 = grad(r1sum)
        @test check_grads(r0,a1)
        @test check_grads(r0,t1)
        @test check_grads(r0,d1)
        @test check_grads(r1sum,a1)
        @test check_grads(r1sum,t1)
        @test check_grads(r1sum,d1)

    end

    @testset "view" begin
        a = rand(3,3)    
        g1 = grad(x->sum(x[1:2,1:2].^2.5))
        g2 = grad(x->sum(view(x,1:2,1:2).^2.5))
        @test g1(a) == g2(a)

        g1 = grad(x->sum(x[:,1:2].^2.0))
        g2 = grad(x->sum(view(x,:,1:2).^2.0))
        g3 = grad(x->sum(selectdim(x,2,1:2).^2.0))
        @test g1(a) == g2(a) == g3(a)
    end

end

nothing
