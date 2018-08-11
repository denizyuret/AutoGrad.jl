using AutoGrad, Test

# TODO: merge gradcheck and check_grads.
# gradcheck iterates over the elements of the first arg.
# check_grads constructs numerical gradient of all args then compares.
# gradcheck has the ability to sample large arrays, check_grads cannot.
# check_grads can handle Tuples and Dicts, gradcheck cannot.
# gradcheck handles non-scalar functions turning them into scalars.

"""

    gradcheck(f, w, x...; kwargs...)

Numerically check the gradient of `f(w,x...;o...)` with respect to its
first argument `w` and return a boolean result.

The argument `w` can be a Number, Array, Tuple or Dict which in turn
can contain other Arrays etc.  Only the largest 10 entries in each
numerical gradient array are checked by default.  If the output of f
is not a number, gradcheck constructs and checks a scalar function by
taking its dot product with a random vector.

# Keywords

* `gcheck=10`: number of largest entries from each numeric array in
  gradient `dw=(grad(f))(w,x...;o...)` compared to their numerical
  estimates.

* `verbose=false`: print detailed messages if true.

* `kwargs=[]`: keyword arguments to be passed to `f`.

* `delta=atol=rtol=cbrt(eps(w))`: tolerance parameters.  See
  `isapprox` for their meaning.

"""
function gradcheck(f, w, x...; kwargs=[], o...)
    y = f(w, x...; kwargs...)
    if !isa(y,Number); f = gc_scalar(f); end
    g = grad(f)
    d = g(w, x...; kwargs...)
    if isa(w, Number)
        gc_number(d, f, w, x...; kwargs=kwargs, o...)
    elseif isbitstype(eltype(w))
        gc_array(w, d, f, w, x...; kwargs=kwargs, o...)
    else
        k = gc_indices(w)
        pass = true
        for i in k
            pass &= gc_index(w, d, i, f, w, x...; kwargs=kwargs, o...)
        end
        return pass
    end
end

function gc_number(d, f, w, x...; delta=gc_dx(w),rtol=gc_dx(w),atol=gc_dx(w),verbose=false,kwargs=[])
    (w1, w2) = gc_interval(w, delta)
    (f1, f2) = (f(w1,x...;kwargs...), f(w2,x...;kwargs...))
    nd = (f2-f1) / (w2-w1)
    di = (d===nothing ? zero(nd) : d)
    if !isapprox(di, nd; rtol=rtol, atol=atol)
        if verbose; @warn("d=$d nd=$nd"); end
        return false
    else
        if verbose && (d*nd!=0); println("gcheck: d=$d nd=$nd"); end
        return true
    end
end

function gc_index(w, d, i, f, w0, x...; o...)
    di = nothing
    try; di = d[i]; catch; end
    if isa(w[i], Number)
        gc_array(w, d, f, w0, x...; icheck=i, o...)
    elseif isbitstype(eltype(w[i]))
        gc_array(w[i], di, f, w0, x...; o...)
    else
        k = gc_indices(w[i])
        pass = true
        for j in k
            pass &= gc_index(w[i], di, j, f, w0, x...; o...)
        end
        return pass
    end
end

# TODO: handle Tuples, Dict

function gc_array(w, d, f, worig, x...; gcheck=10, icheck=0, kwargs=[],
                  delta=0, atol=0, rtol=0, verbose=false)
    if icheck > 0
        irange = (icheck:icheck)
    elseif length(w) <= gcheck
        irange = (1:length(w))
    else # if d == nothing
        irange = rand(1:length(w), gcheck)
    #else
    #   irange = sortperm(abs(vec(Array(d))),rev=true)[1:gcheck]
    end
    wi = w[irange[1]]
    if delta == 0; delta = gc_dx(wi); end
    if atol == 0; atol = gc_dx(wi); end
    if rtol == 0; rtol = gc_dx(wi); end
    pass = true
    for i in irange
        w0 = w[i]
        (w1, w2) = gc_interval(w0, delta)
        w[i] = w1
        f1 = f(worig, x...; kwargs...)
        w[i] = w2
        f2 = f(worig, x...; kwargs...)
        w[i] = w0
        nd = (f2-f1) / (w2-w1)
        di = (d===nothing ? zero(nd) : d[i])
        if !isapprox(di, nd; rtol=rtol, atol=atol)
            if verbose; @warn("d=$di nd=$nd"); end
            pass = false
        else
            if verbose && (di*nd!=0); println("gcheck: d=$di nd=$nd"); end
        end
    end
    return pass
end

gc_dx(x::Number)=cbrt(eps(float(x)))
gc_dx(x)=cbrt(eps(float(eltype(x))))
gc_indices(w::Tuple)=(1:length(w))
gc_indices(w)=eachindex(w)

function gc_interval(w,d)
    w1=w-d/2
    w2=w+d/2
    (w1 < 0 < w) && (w1=zero(w))
    (w2 > 0 > w) && (w2=zero(w))
    return (w1,w2)
end

function gc_scalar(f)
    # r = MersenneTwister(0)
    function g(x...; o...)
        try
            y = f(x...; o...)
            # v = getval(y)
            # srand(r,1)
            # a = oftype(v, rand(r, size(v)))
            # return sum(y .* a)
            if isa(getval(y), AbstractDict)
                return sumvalues(y)
            else
                return sum(y)  # TODO: revert this back to y.*a once julia6 compat issues resolved?
            end
        catch e
            @warn "Cannot convert `$f` to a scalar function: $e" maxlog=1
            return 0
        end
    end
    return g
end


### Testing Utilities:

#if !(@isdefined addtest)
let tests=[]
    global addtest,runtest,alltests
    alltests()=tests
    addtest(t...)=push!(tests,t)
    function runtest(a=tests)   # 20180725: rename runtests->runtest to avoid Base conflict
        for fx in a
            try 
                # tx = fixtest(fx...)
                # check_grads(tx...; fname=fx[1]) || throw(:fail)
                f = eval(AutoGrad,fx[1])
                x = fx[2:end]
                gradcheck(f,x...) || throw(:fail)
            catch e
                @warn((fx...,"$e"))
            end
        end
    end
end
#end

# gradcheck only checks the first arg, this helper will allow us to check all args

applyN(x,f)=f(x...)
addtestN(f,x...)=addtest(:applyN,map(y->Core.eval(AutoGrad,y), collect(Any,x)),Core.eval(AutoGrad,f))
gradcheckN(f,x...;o...)=gradcheck(applyN,collect(Any,x),f;o...)

# Generate tests based on given ranges

function addtest1(f,r=(-Inf,Inf))          # unary
#    bf = broadcast_func(f)
    addtest(f,randin(r))
#    addtest(bf,randin(r,2))
end

function addtest2(f,r1=(-Inf,Inf),r2=r1)   # binary
    # bf = broadcast_func(f)
    addtestN(f,randin(r1),randin(r2))
    addtestN(broadcast,f,randin(r1),randin(r2,2))
    addtestN(broadcast,f,randin(r1,2),randin(r2))
    addtestN(broadcast,f,randin(r1,2),randin(r2,2))
end

function randin(range, dims...; eps=0.01)
    if isa(range, UnitRange{Int})
        rand(range, dims...)
    elseif range==(-Inf,Inf)
        r = randn(dims...)
        sign.(r)*eps + r
    elseif range==(0,Inf)
        eps .- log.(rand(dims...))
    elseif range==(1,Inf)
        eps + 1 .- log.(rand(dims...))
    elseif range==(-1,Inf)
        eps - 1 .-log.(rand(dims...))
    elseif range==(-1,1)
        (1-eps)*(2rand(dims...)-1)
    elseif range==(0,1)
        eps+(1-2eps)*rand(dims...)
    elseif range==(0,2)
        eps+2*(1-eps)*rand(dims...)
    elseif range==(-Inf,-1,1,Inf)
        x = sec.(randn(dims...))
        sign.(x)*eps + x
    else
        error("Unknown range $range")
    end
end

# Alternative gradient check utility -- deprecated.

# EPS, RTOL, ATOL = 1e-4, 1e-4, 1e-6
EPS, RTOL, ATOL = 1e-4, 1e-2, 1e-4

# Check the computed gradients for fun(args) comparing them with numeric
# approximations.  Deprecated, use `gradcheck` instead.

function check_grads(fun, args...; eps=EPS, rtol=RTOL, atol=ATOL, fname=fun)
    #@dbg 2 (:check_grads,fname,:args,args...)
    isempty(args) && error("No args given")
    exact = ntuple(i->grad(fun,i)(args...), length(args))
    numeric = nd(fun, args...; eps=eps)
    #@dbg 2 (:check_grads,fname,:exact,exact,:numeric,numeric)
    same = isequivalent(exact, numeric; rtol=rtol, atol=atol)
    #same || @warn((:check_grads,fname,:args,args,:exact,exact,:numeric,numeric))
    return same
end

function nd(f, args...; eps=EPS)
    #@dbg 2 (:nd,f,args..., :eps, eps)
    unary_f = x->f(x...)
    unary_nd(unary_f, args, eps)
end

unary_nd(f, x::Tuple, eps)         = ntuple(i->unary_nd(indexed_function(f, x, i), x[i], eps), length(x))
unary_nd(f, x::AbstractDict, eps)  = (a=empty(x); for(k,v) in x; a[k] = unary_nd(indexed_function(f, x, k), v, eps); end; a)
unary_nd(f, x::AbstractArray, eps) = reshape(eltype(x)[unary_nd(indexed_function(f, x, i), v, eps) for (i,v) in enumerate(x)], size(x))
unary_nd(f, x::Complex, eps)       = ((f(x + eps/2) - f(x - eps/2)) / eps - im*(f(x + im*eps/2) - f(x - im*eps/2)) / eps)
unary_nd(f, x::Real, eps)          = ((f(x + eps/2) - f(x - eps/2)) / eps)

function indexed_function(fun, arg, index)
    function partial_function(x)
        if isa(arg, Tuple)
            local_arg = (arg[1:index-1]..., x, arg[index+1:end]...)
        else
            local_arg = copy(arg); local_arg[index] = x
        end
        return fun(local_arg)
    end
    return partial_function
end

# isequivalent uses isapprox for Number and AbstractArray{T<:Number}
isequivalent(x::Number,y::Number; o...)=isapprox(x,y;o...)
isequivalent(x::AbstractArray{T},y::AbstractArray{S}; o...) where {T<:Number,S<:Number} = (size(x)==size(y) && isapprox(x,y;o...))

# isequivalent extends to Tuple, AbstractDict, and other Arrays, comparing elementwise
isequivalent(x::Tuple, y::Tuple; o...)=(length(x)==length(y) && all(i->isequivalent(x[i],y[i];o...), 1:length(x)))
isequivalent(x::AbstractArray, y::AbstractArray; o...)=(length(x)==length(y) && all(i->isequivalent(x[i],y[i];o...), 1:length(x)))
isequivalent(x::AbstractDict, y::AbstractDict; o...)=all(k->isequivalent(get(x,k,nothing),get(y,k,nothing);o...), unique([keys(x)...,keys(y)...]))

# isequivalent treats `nothing` as equivalent to zero or zero array.
isequivalent(x::Number,z::Nothing; o...)=isequivalent(z,x;o...)
isequivalent(x::AbstractArray{T},z::Nothing; o...) where T<:Number = isequivalent(z,x;o...)
isequivalent(z::Nothing,x::Number; o...)=isapprox(zero(x),x;o...)
isequivalent(z::Nothing,x::AbstractArray{T}; rtol::Real=Base.rtoldefault(T), atol::Real=0, norm::Function=vecnorm) where T<:Number = (norm(x) <= atol/(1-rtol)) # Modified from: linalg/generic.jl:522

function fixtest(f, x...)
    f = eval(f)
    y = f(x...)
    # detect and prevent testing of zero / undefined grads
    plist = Any[]               # define fnew(plist)
    alist = Any[x...]           # to return f(alist)
    fargs = Any[]               # call fnew(fargs...)
    for i=1:length(alist)
        gargs = Any[Grad{i},y,y,x...]
        gargs[i+3] = Rec(gargs[i+3])
        g = nothing
        try
            g = f(gargs...)
        catch e
            if isa(e,MethodError) && e.f === f && e.args[1] === Grad{i}
                continue        # @warn("No grad $i for $f: $e")
            else
                error("Error during $f$((gargs...,)): $e")
            end
        end
        g === nothing && continue # zero grads
        push!(fargs, alist[i])
        alist[i] = Symbol("x$i")
        push!(plist, alist[i])
    end
    isempty(fargs) && error("$f has no differentiable arguments.")
    f1=f; f = eval(Expr(:->, Expr(:tuple, plist...), Expr(:call, f1, alist...)))
    # if f has non-scalar output, sum it
    isbits(y) || (f2=f; f=(x...)->toscalar(f2(x...)))
    return (f,fargs...)
end

function toscalar(xv; rng=MersenneTwister(0))
    x = getval(xv)
    isa(x,Number) && return xv
    isa(x,UngetIndex) && (x = full(x))
    idx = isa(x,Tuple) ? (1:length(x)) : eachindex(x)
    s = 0
    for i in idx
        s += xv[i] * rand(rng)
    end
    return s
end

# sumvalues sums values of dictionaries, otherwise acts like sum:

import .Broadcast: broadcasted, broadcast # TODO: this should not be required for using @primitive
sumvalues(x)=sum(x)
sumvalues(x::AbstractDict)=sum(values(x))
@primitive sumvalues(x::AbstractDict),ds fillvalues(ds,x)
fillvalues(v,x)=(y=empty(x);for k in keys(x); y[k]=v; end; y)
@primitive fillvalues(v,x),dxv sumvalues(dxv) nothing
# addtest(:sumvalues, Dict(1=>1.,2=>2.))
# addtest(:fillvalues, 0., Dict(1=>1.,2=>2.,3=>3.))

"Test a numeric function with randn scalars and randn arrays, possibly transforming the input to match the domain"
function randcheck(f,t1=identity,ts...; xargs=())
    f1(xs)=f(xs..., xargs...)
    f2(xs)=f.(xs..., xargs...)
    ts = [t1,ts...]
    x1 = map(t->t(randn()), ts)
    x2 = map(t->t.(randn(2)), ts)
    gradcheck(f1, x1) && gradcheck(f2, x2)
end

nothing
