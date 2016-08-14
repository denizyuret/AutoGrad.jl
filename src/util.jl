# I would like to make these type signatures as specific as possible.
# The following are not allowed yet, see https://github.com/JuliaLang/julia/issues/3766
# f{T<:Number,A<:AbstractArray{T}}(x::Node{A})
# f{T<:Number,A<:AbstractArray}(x::Node{A{T}})

function defgrads(grads::Dict{Symbol,Any}, argtypes...)
    for (_f,_d) in grads
        fsig = addtypes(:($_f{}()), argtypes...)
        if _d == :todo
            continue
        elseif _d == 0
            @eval @zerograd $fsig # This defines gradient=0 for all args
        else
            @eval @primitive $fsig
            for i=1:length(argtypes)
                gsig = addtypes(:($_f{}(::Type{Grad{$i}},y::Node)), argtypes...)
                if length(argtypes) == 1
                    @eval $gsig=(dy->dy.*$_d)
                else
                    xi = symbol(:x,i)
                    @eval $gsig=unbroadcast(y, $xi, (dy->dy.*$(_d[i])))
                end
            end
        end
    end
end

typealias Fn{F} Type{Val{F}}

function addtypes(ex::Expr, types...)
    # construct method signature
    # example input: :(exp{}()), Number, Number
    # example output: exp{T1<:Number,T2<:Number}(x1::Union{Node{T1<:Number},T1<:Number}, x2::Union{Node{T2<:Number},T2<:Number})
    if length(types) == 0
        error("Need argument types")
    elseif length(types) == 1
        push!(ex.args[1].args, Expr(:<:, :T, types[1]))
        push!(ex.args, :(x::Node{T}))
    else
        for i=1:length(types)
            Ti = symbol(:T,i); xi = symbol(:x,i)
            push!(ex.args[1].args, Expr(:<:, Ti, types[i]))
            push!(ex.args, :($xi::Nval{$Ti}))
        end
    end
    ex
end

function testgrads(grads::Dict{Symbol,Any}, argtypes...)
    for (_f,_d) in grads
        _d == :todo && continue
        feval = eval(_f)
        ftest(x...)=sum(feval(x...))
        name(ftest,(:sum,_f))   # for debug output
        test = testargs(Val{_f}, argtypes...) # so we can handle functions like acos with restricted domains
        try 
            check_grads(ftest, test...)
        catch e
            println(e)
        end
    end
end

function testargs(f, a...)
    ntuple(length(a)) do i
        a[i] <: Number ? randn() :
        a[i] <: AbstractArray ? randn(2) :
        error("testargs: $(a[i])")
    end
end

EPS, RTOL, ATOL = 1e-4, 1e-4, 1e-6

function check_grads(fun, args...; eps=EPS, rtol=RTOL, atol=ATOL)
    dbg(:cfun,name(fun))
    dbg(:check_grads,(name(fun),:args,args...))
    isempty(args) && error("No args given")
    exact = ntuple(i->grad(fun,i)(args...), length(args))
    numeric = nd(fun, args...; eps=eps)
    dbg(:check_grads,(name(fun),:exact,exact,:numeric,numeric))
    same = isapprox(exact, numeric; rtol=rtol, atol=atol)
    same || warn((:check_grads,name(fun),:args,args,:exact,exact,:numeric,numeric))
    return same
end

function nd(f, args...; eps=EPS)
    dbg(:nd,(f,args..., :eps, eps))
    unary_f = x->f(x...)
    unary_nd(unary_f, float(args), eps)
end

unary_nd(f, x::Tuple, eps)         = ntuple(i->unary_nd(indexed_function(f, x, i), x[i], eps), length(x))
unary_nd(f, x::Associative, eps)   = [k => unary_nd(indexed_function(f, x, k), v, eps) for (k,v) in x]
unary_nd(f, x::AbstractArray, eps) = reshape(eltype(x)[unary_nd(indexed_function(f, x, i), v, eps) for (i,v) in enumerate(x)], size(x))
unary_nd(f, x::Complex, eps)       = ((f(x + eps/2) - f(x - eps/2)) / eps - im*(f(x + im*eps/2) - f(x - im*eps/2)) / eps)
unary_nd(f, x::Real, eps)          = ((f(x + eps/2) - f(x - eps/2)) / eps)

function indexed_function(fun, arg, index)
    function partial_function(x)
        local_arg = copy(arg)
        if isa(local_arg, Tuple)
            local_arg = (local_arg[1:index-1]..., x, local_arg[index+1:end]...)
        else
            local_arg[index] = x
        end
        return fun(local_arg)
    end
    return partial_function
end

# isapprox for Number and AbstractArray{T<:Number} already defined
# extending to Tuple, Associative, and other Arrays

import Base: isapprox
isapprox(x::Tuple, y::Tuple; o...)=(length(x)==length(y) && all(i->isapprox(x[i],y[i];o...), 1:length(x)))
isapprox(x::Associative, y::Associative; o...)=(length(x)==length(y) && all(k->isapprox(x[k],y[k];o...), keys(x)))
isapprox(x::AbstractArray, y::AbstractArray; o...)=(length(x)==length(y) && all(i->isapprox(x[i],y[i];o...), 1:length(x)))

# float for Number and AbstractArray (for isleaftype) already defined
# extend to Tuple, Associative, and arbitrary Arrays

import Base: float
isfloat(x)=isa(x,AbstractFloat)
float(x::Tuple)=(all(isfloat,x) ? x : ntuple(i->float(x[i]), length(x)))
float(x::Associative)=(all(isfloat,values(x)) ? x : [k=>float(v) for (k,v) in x])
float{T<:Number}(x::AbstractArray{T})=reshape([ float(x[i]) for i in eachindex(x) ], size(x))

# The way broadcasting works in Julia:
# y = f(x...) where f is a broadcasting operation.
# size(y) = broadcast_shape(x...)
# ndims(y) = max ndims(x)
# size(y,i) = max size(x,i)
# size(x,i) = 1 or size(y,i) for all x and i<=ndims(x)
# if ndims(x) < ndims(y) the extra dimensions of x are treated as 1

function unbroadcast(ynode, xnode, gradfun)
    x, y = getval(xnode), getval(ynode)
    if isa(x, AbstractArray)
        if (size(x)==size(y))
            return gradfun
        else
            function new_fun(dy)
                result = gradfun(dy)
                d = []
                for i=1:ndims(result)
                    size(x,i) == size(result,i) && continue
                    size(x,i) != 1 && throw(DimensionMismatch())
                    push!(d,i)
                end
                return sum(result, d)
            end
            return new_fun
        end
    elseif isa(y, AbstractArray)
        return (dy->sum(gradfun(dy)))
    else
        return gradfun
    end
end
