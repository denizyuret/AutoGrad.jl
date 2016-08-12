EPS, RTOL, ATOL = 1e-4, 1e-4, 1e-6

function check_grads(fun, args...; eps=EPS, rtol=RTOL, atol=ATOL)
    dbg(:check_grads,(name(fun),:args,args...))
    isempty(args) && error("No args given")
    exact = ntuple(i->grad(fun,i)(args...), length(args))
    numeric = nd(fun, args...; eps=eps)
    dbg(:check_grads,(name(fun),:exact,exact,:numeric,numeric))
    isapprox(exact, numeric; rtol=rtol, atol=atol)
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
function float{T}(x::AbstractArray{T})
    if !isleaftype(T)
        reshape([ float(x[i]) for i in eachindex(x) ], size(x))
    else
        convert(AbstractArray{typeof(float(zero(T)))}, x)
    end
end

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
                error("still did not implement unbroadcast completely")
            end
            return new_fun
        end
    elseif isa(y, AbstractArray)
        return (dy->sum(gradfun(dy)))
    else
        return gradfun
    end
end
