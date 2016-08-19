# I would like to make these type signatures as specific as possible.
# The following are not allowed yet, see https://github.com/JuliaLang/julia/issues/3766
# f{T<:Number,A<:AbstractArray{T}}(x::Node{A})
# f{T<:Number,A<:AbstractArray}(x::Node{A{T}})

function defgrads(grads::Dict{Symbol,Any}, argtypes...; dymul=true)
    addtests(grads, argtypes...)
    for (_f,_d) in grads
        fsig = addtypes(:($_f{}()), argtypes...)
        if _d == :todo
            continue
        elseif _d == 0
            @eval @zerograd $fsig # This defines gradient=0 for all args
        else
            @eval @primitive $fsig
            isa(_d,Union{AbstractArray,Tuple}) || (_d = (_d,))
            for i=1:length(_d)  # _d could be shorter than argtypes in which case the other gradients will be undefined
                gsig = addtypes(:($_f{}(::Type{Grad{$i}},y::Node)), argtypes...)
                if _d[i] == 0
                    gexp = 0  # This defines gradient=0 for one arg
                elseif dymul
                    if _d[i] == 1
                        gexp = :identity
                    elseif _d[i] == -1
                        gexp = :-
                    else
                        gexp = :(dy->dy.*$(_d[i]))
                    end
                    if length(_d) > 1
                        xi = Symbol("x$i")
                        gexp = :(unbroadcast(y, $xi, $gexp))
                    end
                else
                    gexp = _d[i]
                end
                @eval $gsig=$gexp
            end
        end
    end
end

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
            Ti = Symbol("T$i"); xi = Symbol("x$i")
            push!(ex.args[1].args, Expr(:<:, Ti, types[i]))
            push!(ex.args, :($xi::Nval{$Ti}))
        end
    end
    ex
end

function addtests(grads::Dict{Symbol,Any}, argtypes...)
    global _tests
    isdefined(:_tests) || (_tests = Any[])
    push!(_tests, (grads, argtypes...))
end

function runtests()
    global _tests
    for test in _tests
        testgrads(test...)
    end
end

function testgrads(grads::Dict{Symbol,Any}, argtypes...)
    for (_f,_d) in grads
        _d == :todo && continue
        f = eval(_f)
        args = testargs(Val{_f}, argtypes...) # so we can handle functions like acos with restricted domains
        # if f has non-scalar output, sum it
        y = f(args...)
        if !isa(y,Number)
            f1 = f
            f = (x...)->sum(f1(x...))
        end
        # detect and prevent testing of zero grads
        if isa(_d,Tuple) && length(_d)>1 && in(0,_d)
            alist = Any[args...]
            plist = Any[]
            args = Any[]
            for i=1:length(_d)
                if _d[i] != 0
                    push!(args, alist[i])
                    alist[i] = Symbol("x$i")
                    push!(plist, alist[i])
                end
            end
            ex = Expr(:->, Expr(:tuple, plist...), Expr(:call, f, alist...))
            f = eval(ex)
        end
        try 
            check_grads(f, args...; fname=_f)
        catch e
            warn((_f,args...,e))
        end
    end
end

function testargs(f, a...)
    @dbgutil((f,a...))
    ntuple(length(a)) do i
        a[i] <: Number ? randn() :
        a[i] <: AbstractArray ? randn(2) :
        error("testargs: $(a[i])")
    end
end

if !isdefined(:Fn)
typealias Fn{F} Type{Val{F}}    # used to create the first argument of testargs
end
Fn2(F)=Type{Val{Symbol("$(F)2")}}   # used for fallback in type specific testargs

EPS, RTOL, ATOL = 1e-4, 1e-4, 1e-6

# TODO: do sampling or random direction for large args
function check_grads(fun, args...; eps=EPS, rtol=RTOL, atol=ATOL, fname=fun)
    @dbgutil((:check_grads,fname,:args,args...))
    isempty(args) && error("No args given")
    exact = ntuple(i->grad(fun,i)(args...), length(args))
    numeric = nd(fun, args...; eps=eps)
    @dbgutil((:check_grads,fname,:exact,exact,:numeric,numeric))
    same = isequivalent(exact, numeric; rtol=rtol, atol=atol)
    same || warn((:check_grads,fname,:args,args,:exact,exact,:numeric,numeric))
    return same
end

function nd(f, args...; eps=EPS)
    @dbgutil((:nd,f,args..., :eps, eps))
    unary_f = x->f(x...)
    unary_nd(unary_f, tofloat(args), eps)
end

unary_nd(f, x::Tuple, eps)         = ntuple(i->unary_nd(indexed_function(f, x, i), x[i], eps), length(x))
unary_nd(f, x::Associative, eps)   = (a=similar(x); for(k,v) in x; a[k] = unary_nd(indexed_function(f, x, k), v, eps); end; a)
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
isequivalent{T<:Number,S<:Number}(x::AbstractArray{T},y::AbstractArray{S}; o...)=isapprox(x,y;o...)

# isequivalent extends to Tuple, Associative, and other Arrays, comparing elementwise
isequivalent(x::Tuple, y::Tuple; o...)=(length(x)==length(y) && all(i->isequivalent(x[i],y[i];o...), 1:length(x)))
isequivalent(x::Associative, y::Associative; o...)=(length(x)==length(y) && all(k->isequivalent(x[k],y[k];o...), keys(x)))
isequivalent(x::AbstractArray, y::AbstractArray; o...)=(length(x)==length(y) && all(i->isequivalent(x[i],y[i];o...), 1:length(x)))

# isequivalent treats `nothing` as equivalent to zero or zero array.
isequivalent(x::Number,z::Void; o...)=isequivalent(z,x;o...)
isequivalent{T<:Number}(x::AbstractArray{T},z::Void; o...)=isequivalent(z,x;o...)
isequivalent(z::Void,x::Number; o...)=isapprox(zero(x),x;o...)
isequivalent{T<:Number}(z::Void,x::AbstractArray{T}; rtol::Real=Base.rtoldefault(T), atol::Real=0, norm::Function=vecnorm) = (norm(x) <= atol/(1-rtol)) # Modified from: linalg/generic.jl:522

# TODO: check if we really need tofloat.
# tofloat uses float for Number and AbstractArray (for isleaftype)
tofloat(x::Number)=float(x)
tofloat{T<:Number}(x::AbstractArray{T})=float(x)

# tofloat extends to Tuple, Associative, and arbitrary Arrays
tofloat(x::Tuple)=(all(isfloat,x) ? x : ntuple(i->tofloat(x[i]), length(x)))
tofloat(x::Associative)=(all(isfloat,values(x)) ? x : (a=similar(x); for (k,v) in x; a[k]=tofloat(v); end; a))
tofloat(x::AbstractArray)=(all(isfloat,x) ? x : map(tofloat,x))
isfloat(x)=isa(x,AbstractFloat)

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

# Pretty print for debugging:
# TODO: replace this with dbgprint, prevents gc()!
_name=ObjectIdDict()
name(f,n)=(_name[f]=n)
name(f)=get(_name,f,f)
name(x::ReverseNode)=Symbol("R$(href(x))")
name(x::Node)=Symbol("N$(href(x))")
name(x::Array)=Symbol("A$(join([href(Ref(x)),size(x)...],'x'))")
name(x::Tuple)=map(name,x)
href(x)=Int(hash(x)%100)

Base.show(io::IO, n::Node) = print(io,"$(name(n))$((name(n.value),[(name(t),name(r)) for (t,r) in n.tapes]...))")
Base.show(io::IO, n::ReverseNode) = print(io,"$(name(n))$((name(n.node.value),map(name,n.outgrads),[(name(y),name(x)) for (x,y) in n.parent_grad_ops]...))")
