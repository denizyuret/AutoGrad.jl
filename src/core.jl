# Set TIMER=true if you want profiling information in AutoGrad.to
TIMER=false
using TimerOutputs; const to = TimerOutput()
macro timeforw(expr); (TIMER ? :(@timeit to $(esc(:(f==broadcast ? "$(argvals[1])." : "$f"))) $(esc(expr))) : esc(expr)); end
macro timeback(expr); (TIMER ? :(@timeit to $(esc(:(r.func==broadcast ? "$(r.args[1]).[$(i-1)]" : "$(r.func)[$i]"))) $(esc(expr))) : esc(expr)); end

abstract type Value{T} end

mutable struct Param{T} <: Value{T}
    value::T
    opt
    Param(x::T) where T = new{T}(x)
end

mutable struct Result{T} <: Value{T}
    value::T
    func::Function
    args::Tuple
    kwargs::Base.Iterators.Pairs
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

grad(t,x)=nothing
grad(t::Tape,x::Value)=(n=get(t,x,nothing); n===nothing ? n : n.outgrad)

value(x)=x
value(x::Value)=x.value
value(x::Tape)=first(x).Value.value

_tapes = Tape[]

abstract type Arg{N} end

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
            @timeback g = back(r.func, Arg{i}, n.outgrad, r, r.args...; r.kwargs...)
            p.outgrad = sum_outgrads(p.outgrad, g)
        end
        if isempty(_tapes) && isa(r,Result); n.outgrad = nothing; end  # saves memory
    end
    return tape
end

# This allows argument expressions like @diff sin(sqrt(x)) which fail with differentiate
# because arguments get evaluated before the tape gets created.
macro diff(fx); :(differentiate(()->$(esc(fx)))); end

duplicate(x)=(isa(x,Value) ? identity(x) : x)

back(x...; o...) = nothing

function forw(f, args...; kwargs...)
    argvals = value.(args)
    @timeforw result = f(argvals...; kwargs...)
    if isempty(_tapes); return result; end
    result = Result(result, f, args, kwargs)
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

function grad(fun::Function, argnum::Int=1, loss=false)
    function gradfun(args...; kwargs...)
        arg_wrt = args[argnum]
        if !isa(arg_wrt,Value); arg_wrt = Param(arg_wrt); end
        args = Any[args...]
        args[argnum] = arg_wrt
        result = differentiate(fun, args...; kwargs...)
        xgrad = isa(result, Tape) ? last(result).outgrad : nothing
        return loss ? (xgrad,value(result)) : xgrad
    end
    return gradfun
end

gradloss(f,a=1)=grad(f,a,true)
