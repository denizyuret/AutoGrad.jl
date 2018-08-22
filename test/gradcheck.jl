using AutoGrad: Tape, Rec, Node, findeq, complete!, sum_outgrads, back

"""

    gradcheck(f, x...; kwargs...)

Numerically check the gradient of `f(x...)` and return a boolean
result.

Each argument can be a Number, Array, Tuple or Dict which in turn can
contain other Arrays etc.  Only 10 random entries in each large
numeric array are checked by default.  If the output of `f` is not a
number, we check the gradient of `sum(f(x...))`.

# Keywords

* `args=:`: the argument indices to check gradients with respect
  to. Could be an array or range of indices or a single index. By
  default all arguments that have a `length` method are checked.

* `kw=()`: keyword arguments to be passed to `f`.

* `nsample=10`: number of random entries from each numeric array in
  gradient `dw=(grad(f))(w,x...;o...)` compared to their numerical
  estimates.

* `atol=0.001; rtol=0.01`: tolerance parameters.  See `isapprox` for
  their meaning.

* `delta`: step size for numerical gradient calculation. 
  `cbrt(eps(val))` is used if not specified.

* `verbose=false`: print detailed messages if true.

"""
function gradcheck(f, x...; kw=(), args=:, nsample=10, verbose=1, rtol=0.01, atol=0.001, delta=0.0001)
    args = isa(args, Colon) ? (1:length(x)) : args
    tape = Tape()
    xrec  = Any[x...]
    xnode = Array{Any}(undef, length(x))
    for i in args; (xrec[i],xnode[i]) = gcparam(x[i], tape); end
    result = gcsum(f(xrec...; kw...))
    if isa(result, Rec); gcback(result, tape); end
    f0 = getval(result)
    xptr = Any[x...]
    gptr = Array{Any}(undef, length(x))
    for i in args; gptr[i] = xnode[i].outgrad; end
    all(args) do i
        gcwalk(i, xptr, gptr, f0, f, xptr, kw, nsample, verbose, delta, rtol, atol)
    end
end

function gcwalk(i, xptr, gptr, f0, f, x, kw, nsample, verbose, delta, rtol, atol)
    if isa(xptr[i], Number)
        xorig = xptr[i]
        delta = delta > 0 ? delta : cbrt(eps(xorig))
        xptr[i] = xorig >= 0 ? xorig + delta : xorig - delta
        f1 = gcsum(f(x...; kw...))
        nd = (f1 - f0) / (xptr[i] - xorig)
        xptr[i] = xorig
        ad = gcget(gptr,i,0)
        result = isapprox(nd, ad, rtol=rtol, atol=atol)
        if verbose > 1 || (!result && verbose > 0); @show xorig,f0,nd,ad; end
        return result
    elseif !isempty(methods(length,(typeof(xptr[i]),)))
        n = length(xptr[i])
        if isa(xptr[i], Tuple)
            xptr[i] = Any[xptr[i]...]
            indices = 1:n
        elseif isa(xptr[i], Dict)
            indices = keys(xptr[i])
        elseif isbitstype(eltype(xptr[i]))
            indices = n <= nsample ? (1:n) : rand(1:n, nsample)
        else
            indices = 1:n
        end
        return all(indices) do j
            gcwalk(j, xptr[i], gcget(gptr,i,nothing), f0, f, x, kw, nsample, verbose, delta, rtol, atol)
        end
    end
end

function gcparam(x,tape)
    if isa(x,Rec)
        x = identity(x)
        n = Node(x,tape)
    else
        x = Rec(x,tape)
        n = x.nodes[1]
    end
    return x, n
end

function gcback(y, tape)
    tapeidx = findeq(y.tapes, tape)
    y.nodes[tapeidx].outgrad = one(y.value)
    complete!(tape)
    for n in tape[end-1:-1:1]
        if n.outgrad === nothing; continue; end
        r = n.rec
        for i in 1:length(n.parents)
            if !isassigned(n.parents,i); continue; end
            p = n.parents[i]
            b = back(r.func, Val(i), n.outgrad, r, r.args...; r.kwargs...)
            p.outgrad = sum_outgrads(p.outgrad, b)
        end
    end
end

function gcget(g,i,default=nothing)
    if isa(g,Nothing)
        return default
    elseif isa(g,AbstractDict) && !haskey(g,i)
        return default
    elseif isa(g,AbstractArray) && !isassigned(g,i)
        return default
    elseif isa(g,Tuple) && !in(i,1:length(g))
        return default
    elseif isa(g[i],Nothing)
        return default
    end
    return g[i]
end

function gcsum(x)
    v = getval(x)
    if isa(v,Number)
        return x
    elseif isempty(v)
        return 0
    else
        return sum(x)
    end
end

"Test a numeric function with randn scalars and randn arrays, possibly transforming the input to match the domain"
function randcheck(f,t1=identity,ts...; args=:, kw...)
    ts = [t1,ts...]
    if isa(args,Colon); args=1:length(ts); end
    x64 = map(t->t(randn()), ts)
    a64 = map(t->t.(randn(2)), ts)
    gradcheck(f, x64...; args=args, kw...) &&
    gradcheck(broadcast, f, a64...; args=(args.+1), kw...)
    # Uncomment to test with Float32, much harder to pass tests
    # x32 = map(t->t(randn(Float32)), ts)
    # a32 = map(t->t.(randn(Float32,2)), ts)
    # gradcheck(f, x32...; args=args, kw...) &&
    # gradcheck(broadcast, f, a32...; args=(args.+1), kw...) &&
end

ϵ = 0.1
abs_gt_0(x)=(x < -ϵ ? x : x < 0 ? -ϵ : x < ϵ ? ϵ : x)
abs_lt_1(x)=rand()*(2-2ϵ)-(1-ϵ)
abs_gt_1(x)=1/abs_lt_1(x)
val_lt_1(x)=clamp(rand(),ϵ,1-ϵ)
val_lt_2(x)=clamp(2*rand(),ϵ,2-ϵ)
val_gt_1(x)=1/val_lt_1(x)
val_gt_0(x)=clamp(abs(x),ϵ,Inf)

