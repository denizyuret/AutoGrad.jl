using AutoGrad

"""

    gradcheck(f, x...; kwargs...)

Numerically check the gradient of `f(x...)` and return a boolean
result.

Each argument can be a Number, Array, Tuple or Dict which in turn can contain other Arrays
etc.  Only 10 random entries in each large numeric array are checked by default.  If the
output of `f` is not a number, we check the gradient of `sum(f(x...))`. See also `gcheck`
for a different take on marking parameters.

# Keywords

* `args=:`: the argument indices to check gradients with respect
  to. Could be an array or range of indices or a single index. By
  default all arguments that have a `length` method are checked.

* `kw=()`: keyword arguments to be passed to `f`.

* `nsample=10`: number of random entries from each numeric array in
  gradient `dw=(grad(f))(w,x...;o...)` compared to their numerical
  estimates.

* `atol=rtol=0.01`: tolerance parameters.  See `isapprox` for
  their meaning.

* `delta=0.0001`: step size for numerical gradient calculation.

* `verbose=1`: 0 prints nothing, 1 shows failing tests, 2 shows all tests.

"""
function gradcheck(f, x...; kw=(), args=:, nsample=10, verbose=1, rtol=0.05, atol=0.01, delta=0.0001)
    args = isa(args, Colon) ? (1:length(x)) : args
    xrec  = Any[x...]
    for i in args; xrec[i] = Param(xrec[i]); end
    result = @diff gcsum(f, xrec...; kw...)
    f0 = value(result)
    xptr = Any[x...]
    gptr = Array{Any}(undef, length(x))
    for i in args; gptr[i] = full(grad(result, xrec[i])); end
    all(args) do i
        gcwalk(i, xptr, gptr, f0, f, xptr, kw, nsample, verbose, delta, rtol, atol)
    end
end

function gcsum(f,x...;o...)
    y = f(x...;o...)
    v = value(y)
    if isa(v,Number)
        return y
    elseif isempty(v)
        return 0
    else
        return sum(y)
    end
end

function gcwalk(i, xptr, gptr, f0, f, x, kw, nsample, verbose, delta, rtol, atol)
    if isa(value(xptr[i]), Number)
        if isa(xptr[i], Param)
            xi = xptr[i].value
            delta = delta > 0 ? delta : cbrt(eps(xi))
            xptr[i].value = xi >= 0 ? xi + delta : xi - delta
            f1 = gcsum(f, x...; kw...)
            nd = (f1 - f0) / (xptr[i] - xi)
            xptr[i].value = xi
        else
            xi = xptr[i]
            delta = delta > 0 ? delta : cbrt(eps(xi))
            xptr[i] = xi >= 0 ? xi + delta : xi - delta
            f1 = gcsum(f, x...; kw...)
            nd = (f1 - f0) / (xptr[i] - xi)
            xptr[i] = xi
        end
        ad = gcget(gptr,i,0)
        result = isapprox(nd, ad, rtol=rtol, atol=atol)
        if verbose >= 2 || (!result && verbose >= 1)
            #fn = (f==broadcast ? x[1] : f)
            pa = summary(xptr)
            @show pa,xi,f0,nd,ad
        end
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
    else
        return true
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

"Test a numeric function with Float32/64 randn scalars and randn arrays, possibly transforming the input to match the domain"
function randcheck(f,t1=identity,ts...; args=:, kw...)
    ts = [t1,ts...]
    if isa(args,Colon); args=1:length(ts); end
    x64 = map(t->t(randn()), ts)
    a64 = map(t->t.(randn(2)), ts)
    x32 = map(t->t(randn(Float32)), ts)
    a32 = map(t->t.(randn(Float32,2)), ts)
    gradcheck(f, x32...; args=args, kw...) &&
    gradcheck(broadcast, f, a32...; args=(args.+1), kw...) &&
    gradcheck(f, x64...; args=args, kw...) &&
    gradcheck(broadcast, f, a64...; args=(args.+1), kw...)
end


"""

    gcheck(f, x...; kw, o...)
    @gcheck f(x...; kw...) (opt1=val1,opt2=val2,...)

Numerically check the gradient of `f(x...; kw...)` and return a boolean result.

Example call: `gcheck(nll,model,x,y)` or `@gcheck nll(model,x,y)`. The parameters should be
marked as `Param` arrays in `f`, `x`, and/or `kw`.  Only 10 random entries in each large
numeric array are checked by default.  If the output of `f` is not a number, we check the
gradient of `sum(f(x...; kw...))`. Keyword arguments:

* `kw=()`: keyword arguments to be passed to `f`, i.e. `f(x...; kw...)`
* `nsample=10`: number of random entries from each param to check
* `atol=0.01,rtol=0.05`: tolerance parameters.  See `isapprox` for their meaning.
* `delta=0.0001`: step size for numerical gradient calculation.
* `verbose=1`: 0 prints nothing, 1 shows failing tests, 2 shows all tests.

"""
gcheck, :(@gcheck)

function gcheck(f, x...; kw=(), nsample=10, verbose=1, rtol=0.05, atol=0.01, delta=0.0001)
    y = @diff gcsum(f, x...; kw...)
    if !isa(y, Tape); @warn("Output independent of params"); return true; end
    f0 = value(y)
    ps = Param[ n.Value for n in y.list if isa(n.Value, Param) ]
    if isempty(ps); @error("Cannot find any params"); end
    #vs = value.(ps)
    gs = (p->full(grad(y,p))).(ps)
    all(1:length(ps)) do i
        #gcwalk(i, vs, gs, f0, f, x, kw, nsample, verbose, delta, rtol, atol)
        gcwalk(i, ps, gs, f0, f, x, kw, nsample, verbose, delta, rtol, atol)
    end
end

macro gcheck(fx,options=:(NamedTuple()))
    :(gcheck(()->$(esc(fx));$(esc(options))...))
end
