include("gradcheck.jl")

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

nothing
