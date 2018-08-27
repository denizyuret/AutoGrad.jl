abstract type Value{T} end

mutable struct Param{T} <: Value{T}
    value::T
end

mutable struct Result{T} <: Value{T}
    value::T
    func::Function
    args::Tuple
    kwargs::Base.Iterators.Pairs
    Result(val::T, func, args...; kwargs...) where T = new{T}(val, func, args, kwargs)
end

mutable struct Node
    outgrad
    Value::Value
    parents::Vector{Node}
    cdr::Node
    Node(og,rc,pa,pr)=new(og,rc,pa,pr)
    Node()=new()
end

const Tape = IdDict{Value,Node}
const NIL = Param([])
newtape() = (n=Node(); n.cdr=n; Tape(NIL => n))
Base.iterate(t::Tape,s=(t[NIL],t[NIL])) = ((p,n) = s; p = p.cdr; p === n ? nothing : (p, (p, n)))
Base.collect(t::Tape)=(a=Array{Node}(undef,length(t)-1); i=0; for n in t; a[i+=1]=n; end; a)

gradient(t,x)=nothing
gradient(t::Tape,x::Value)=(n=get(t,x,nothing); n===nothing ? n : n.outgrad)

value(x)=x
value(x::Value)=x.value
value(x::Tape)=first(x).Value.value

_tapes = Tape[]

function differentiate(f, x...; o...)
    global _tapes
    if !isempty(_tapes)       # PR#75: to avoid tape confusion
        x = map(duplicate,x)  # duplicate tracked function arguments.
        o = isempty(o) ? () : pairs(map(duplicate,values(o)))
    end
    tape = newtape()
    push!(_tapes, tape)
    result = nothing
    try
        result = f(x...; o...)
    catch e
        pop!(_tapes); throw(e)
    end
    if pop!(_tapes) !== tape; error("Tape stack error"); end
    if !isa(result,Result); return result; end
    if !isa(value(result), Number); error("Only scalar valued functions supported."); end
    n = first(tape)
    if result !== n.Value; error("Result not on tape"); end
    n.outgrad = one(value(result))
    for n in tape
        if n.outgrad == nothing; continue; end
        r = n.Value
        @inbounds for i in 1:length(n.parents)
            if !isassigned(n.parents, i); continue; end
            p = n.parents[i]
            g = back(r.func, Val(i), n.outgrad, r, r.args...; r.kwargs...)
            p.outgrad = sum_outgrads(p.outgrad, g)
        end
        if isempty(_tapes) && !isa(r,Param); n.outgrad = nothing; end  # saves memory
    end
    return tape
end

duplicate(x)=(isa(x,Value) ? identity(x) : x)

back(::Function, ::Val, dy, y, x...; o...) = nothing

function forw(f, args...; kwargs...)
    argvals = value.(args)
    result = f(argvals...; kwargs...)
    if isempty(_tapes); return result; end
    result = Result(result, f, args...; kwargs...)
    for tape in _tapes
        record(result, tape)
    end
    return result
end

function record(r::Result,t::Tape)
    nargs = length(r.args)
    parents = Array{Node}(undef, nargs)
    @inbounds for argnum = 1:nargs
        arg = r.args[argnum]
        if !isa(arg,Value); continue; end	
        p = get(t,arg,nothing)
        if p === nothing
            p = cons!(arg,t)
        end
        parents[argnum] = p
    end
    cons!(r,t,parents)
end

function cons!(r::Value,t::Tape,parents::Vector{Node}=Node[])
    m = t[NIL]
    n = Node(nothing, r, parents, m.cdr)
    if !isdefined(m,:parents); m.parents = [n]; end
    m.cdr = t[r] = n
end

Base.last(t::Tape)=t[NIL].parents[1] # cons! makes sure this works.

function grad(fun::Function, argnum::Int=1)
    function gradfun(args...; kwargs...)
        arg_wrt = args[argnum]
        if !isa(arg_wrt,Value); arg_wrt = Param(arg_wrt); end
        args = Any[args...]
        args[argnum] = arg_wrt
        result = differentiate(fun, args...; kwargs...)
        isa(result, Tape) ? last(result).outgrad : nothing
    end
    return gradfun
end
