include("header.jl")
# Uncomment these if you want lots of messages:
# import Base.Test: default_handler, Success, Failure, Error
# default_handler(r::Success) = info("$(r.expr)")
# default_handler(r::Failure) = warn("$(r.expr) FAILED")
# default_handler(r::Error)   = warn("$(r.expr): $(r.err)")

# write your own tests here
@test 1 == 1

# Moving from the addtest mechanism to having separate test files here
include("interfaces.jl")

info("Test indexing...")
a1 = rand(3)                    # FAIL: some high-order (b1sum) tests with rand(2)
t1 = (a1...)
d1 = Dict(); for i=1:length(a1); d1[i]=a1[i]; end

s0(x)=x[1]^2+x[2]^2
s1 = grad(s0)
s1sum(x)=(y=s1(x);y[1]+y[2])
s2 = grad(s1sum)
@test check_grads(s0,a1)
@test check_grads(s0,t1)
@test check_grads(s0,d1)
@test_broken check_grads(s1sum,a1)
@test_broken check_grads(s1sum,t1)
@test_broken check_grads(s1sum,d1)

using AutoGrad: sumvalues
f0(x)=(a=0;for i=1:length(x);a+=x[i]^2;end;a)
f1=grad(f0)
f1sum(x)=sumvalues(f1(x))
f2=grad(f1sum)
@test check_grads(f0,a1)
@test check_grads(f0,t1)
@test check_grads(f0,d1)
@test_broken check_grads(f1sum,a1)
@test_broken check_grads(f1sum,t1)
@test_broken check_grads(f1sum,d1)

r0(x)=(s=0; for i=2:length(x); s+=(1-x[i-1])^2 + 100*(x[i]-x[i-1]^2)^2; end; s)
r1 = grad(r0)
r1sum(x)=sumvalues(r1(x))
r2 = grad(r1sum)
@test check_grads(r0,a1)
@test check_grads(r0,t1)
@test check_grads(r0,d1)
@test_broken check_grads(r1sum,a1)
@test_broken check_grads(r1sum,t1)
@test_broken check_grads(r1sum,d1)

info("Test rosenbrock with map...")
b0(x) = sum(map((i, j) -> (1 - j)^2 + 100*(i - j^2)^2, x[2:end], x[1:end-1]))
b1 = grad(b0)
b1sum(x)=sumvalues(b1(x))
@test check_grads(b0,a1)
@test check_grads(b0,t1)
@test_broken check_grads(b1sum,a1) # fail with size 2
@test_broken check_grads(b1sum,t1) # fail with size 2
@time b1sum(rand(10000))

info("Test higher order gradients...")
g1 = grad(sin); @test g1(1)==cos(1)
g2 = grad(g1);  @test g2(1)==-sin(1)
g3 = grad(g2);  @test g3(1)==-cos(1)
g4 = grad(g3);  @test g4(1)==sin(1)
g5 = grad(g4);  @test g5(1)==cos(1)
g6 = grad(g5);  @test g6(1)==-sin(1)
g7 = grad(g6);  @test g7(1)==-cos(1)
g8 = grad(g7);  @test g8(1)==sin(1)
g9 = grad(g8);  @test g9(1)==cos(1)

info("Test neural net...")
n0(w,x,y)=sum(((w[3]*max(0,w[1]*x.+w[2]).+w[4])-y).^2)
n1 = grad(n0)
n1sum(w,x,y)=sum(map(sum,n1(w,x,y)))
n1sumd(w,x,y)=sum(map(sum,values(n1(w,x,y))))
wa = Any[rand(2,3),rand(2),rand(2,2),rand(2)]
wt = (wa...)
wd = Dict(); for i=1:length(wa); wd[i]=wa[i]; end
@test check_grads(n0, wa, rand(3,10), rand(2,10))
@test check_grads(n0, wt, rand(3,10), rand(2,10))
@test check_grads(n0, wd, rand(3,10), rand(2,10))
@test_broken check_grads(n1sum, wa, rand(3,10), rand(2,10))
@test_broken check_grads(n1sum, wt, rand(3,10), rand(2,10))
# TODO: This needs more work:
# @test check_grads(n1sumd, wd, rand(3,10), rand(2,10))  # FAIL

info("Test primitives...")
using AutoGrad: runtests
runtests()



### UTILS...

"Find out where different methods are."
function where(k)
    f = eval(k)
    a = (rand(), rand(2), rand(2,2), rand(2,2,2))
    for x in a
        try println(@which f(x)); end
        for y in a
            try println(@which f(x,y)); end
        end
    end
end

"See which scalar/array args f accepts."
function testargs(k)
    print("$k:")
    f = eval(k)
    try f(rand()); print(" (N,)"); end
    try f(rand(2)); print(" (V,)"); end
    try f(rand(2,2)); print(" (M,)"); end
    try f(rand(2,2,2)); print(" (T,)"); end
    try f(rand(),rand()); print(" (N,N)"); end
    try f(rand(),rand(2)); print(" (N,V)"); end
    try f(rand(),rand(2,2)); print(" (N,M)"); end
    try f(rand(),rand(2,2,2)); print(" (N,T)"); end
    try f(rand(2),rand()); print(" (V,N)"); end
    try f(rand(2),rand(2)); print(" (V,V)"); end
    try f(rand(2),rand(2,2)); print(" (V,M)"); end
    try f(rand(2),rand(2,2,2)); print(" (V,T)"); end
    try f(rand(2,2),rand()); print(" (M,N)"); end
    try f(rand(2,2),rand(2)); print(" (M,V)"); end
    try f(rand(2,2),rand(2,2)); print(" (M,M)"); end
    try f(rand(2,2),rand(2,2,2)); print(" (M,T)"); end
    try f(rand(2,2,2),rand()); print(" (T,N)"); end
    try f(rand(2,2,2),rand(2)); print(" (T,V)"); end
    try f(rand(2,2,2),rand(2,2)); print(" (T,M)"); end
    try f(rand(2,2,2),rand(2,2,2)); print(" (T,T)"); end
    println()
end

function test1arg(x; o...)
    for k in keys(math1arg)
        fx = eval(k)
        f = isa(x,Number) ? fx : a->sum(fx(a))
        (f===fx) || name(f,(:sum,k))
        @test check_grads(f, x; o...)
    end
end

function test2arg(x1,x2,flist=keys(math2arg); o...)
    for k in flist
        fx = eval(k)
        f = isa(x1,Number) && isa(x2,Number) ? fx : (a,b)->sum(fx(a,b))
        (f===fx) || name(f,(:sum,k))
        @test check_grads(f, x1, x2; o...)
    end
end


function test2arg_old(x1,x2;eps=1e-5)
    # First, second, or both arguments can be arrays
    # Grad only works for scalar output, so we'll sum the results
    psum(a)=(isa(a,Number) ? a : (s=0;for i=1:length(a);s+=a[i];end;s))
    for k in keys(math2arg)
        fx = eval(k)
        f = (x1,x2)->psum(fx(x1,x2))
        y = f(x1,x2)
        dbg((k,:x,x1,x2,:y,y))
        g1,g2 = grad(f,1),grad(f,2)
        dx = [g1(x1,x2),g2(x1,x2)]
        # Need to separate here based on array or not, maybe write a gcheck?
        Dx = [(f(x1+eps,x2)-f(x1-eps,x2))/2eps, Dx2 = (f(x1,x2+eps)-f(x1,x2-eps))/2eps]
        println((k,:diff,abs(dx-Dx),:dx,dx,:Dx,Dx))
        isapprox2(dx, Dx) || error("$k derivative is wrong: $((dx,Dx))")
    end
    return true
end

function test2arg_scalar(eps=1e-5)
    for k in keys(math2arg)
        f = eval(k)
        g1,g2 = grad(f,1),grad(f,2)
        x1,x2 = rand(2)
        y = f(x1,x2)
        dbg((k,:x,x1,x2,:y,y))
        dx = [g1(x1,x2),g2(x1,x2)]
        Dx = [(f(x1+eps,x2)-f(x1-eps,x2))/2eps, Dx2 = (f(x1,x2+eps)-f(x1,x2-eps))/2eps]
        println((k,:diff,abs(dx-Dx),:dx,dx,:Dx,Dx))
        # @test_approx_eq_eps dx Dx 1e-6
        isapprox2(dx, Dx) || error("$k derivative is wrong: $((dx,Dx))")
    end
    return true
end


function test1arg_scalar(eps=1e-5)
    for k in keys(math1arg)
        f = eval(k)
        g = grad(f)
        x = rand()
        y = f(x)
        dbg((k,:x,x,:y,y))
        dx = g(x)
        Dx = (f(x+eps)-f(x-eps))/2eps
        println((k,:diff,abs(dx-Dx),:dx,dx,:Dx,Dx))
        # @test_approx_eq_eps dx Dx 1e-6
        isapprox(dx, Dx) || error("$k derivative is wrong: $((dx,Dx))")
    end
    return true
end

function test1arg_array(eps=1e-5)
    for k in keys(math1arg)
        fx = eval(k)
        f = x->(a=fx(x);a[1]+a[2])
        g = grad(f)
        x = rand(2,3)
        y = f(x)
        dbg((k,:x,x,:y,y))
        dx = g(x)
        Dx = similar(x)
        for i=1:length(x)
            xi=x[i]
            x[i]=xi+eps; f1=f(x)
            x[i]=xi-eps; f2=f(x)
            x[i]=xi
            Dx[i]=(f1-f2)/2eps
        end
        println((k,:maxdiff,maximum(abs(dx-Dx)),:avgdx,mean(abs(dx)))) # :dx,dx,:Dx,Dx))
        isapprox2(dx, Dx) || error("$k derivative is wrong: $((dx,Dx))")
    end
    return true
end

function isapprox2(x, y; 
                   maxeps::Real = max(eps(eltype(x)), eps(eltype(y))),
                   rtol::Real=maxeps^(1/3), atol::Real=maxeps^(1/2))
    size(x) == size(y) || (warn("isapprox: $(size(x))!=$(size(y))"); return false)
    x = convert(Array, x)
    y = convert(Array, y)
    d = abs(x-y)
    s = abs(x)+abs(y)
    maximum(d - rtol * s) <= atol
end

# include("math1arg.jl")
# @test test1arg_scalar()
# @test test1arg_array()

function test1(x=(1.,2.))
    # foo(x)=sin(x[1])+cos(x[2])
    foo(x) = x[1]+x[2]
    goo = grad(foo)
    @show goo(x)
    # This does not work because goo is not scalar valued!
    # hoo = grad(goo)
    # @show hoo(x)
end

function test2()
    gsin = grad(sin)
    hsin = grad(gsin)
    #@show sin(1.0)
    #@show gsin(1.0)
    @show hsin(1.0)
end

function test3()
    foo2(x,y)=sin(x)+cos(y)
    goo2 = grad(foo2)
    goo22 = grad(foo2, 2)
    @show goo2(1,2)
    @show goo22(1,2)
end
