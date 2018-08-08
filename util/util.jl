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
