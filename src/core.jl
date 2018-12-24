abstract type Value{T} end

abstract type Tracked{T} <: Value{T} end

mutable struct Param{T} <: Tracked{T}
    value::T; opt
    Param{T}(v,o) where {T} = new(v,o)
    Param{T}(v,o) where {T<:Value} = error("Param cannot take $T as arg.")
end
Param(v::T,o=nothing) = Param{T}(v,o)

mutable struct Result{T} <: Tracked{T}
    value::Union{T,Nothing}     # gcnode sets this to nothing to save memory
    func::Function
    args::Tuple
    kwargs::Base.Iterators.Pairs
    Result{T}(v,f,a,k) where {T} = new(v,f,a,k)
    Result{T}(v,f,a,k) where {T<:Value} = error("Result cannot take $T as arg.")
end
Result(v::T,f,a,k) = Result{T}(v,f,a,k)

mutable struct Bcasted{T} <: Value{T}
    value::T
    Bcasted{T}(v) where {T} = new(v)
    Bcasted{T}(v) where {T<:Bcasted} = v # We do not want Bcasted{Bcasted}
end
Bcasted(v::T) = Bcasted{T}(v)

# Value recursion illegal except Bcasted{<:Tracked}
checktype(x) = true
checktype(x::Value{<:Value}) = false
checktype(x::Bcasted{<:Tracked}) = checktype(x.value)
checktypes(args)=(all(checktype, args) || error("Bad type $(typeof.(args))"))

# value() should give a regular (non-Value) result regardless of recursion
value(x) = x
value(x::Value) = x.value
value(x::Value{<:Value}) = error("Illegal type recursion $(typeof(x))")
value(x::Bcasted{<:Tracked}) = value(x.value)

# To catch whenever one arg is a Value in broadcast expressions, we define a style:
import .Broadcast: BroadcastStyle, Style, broadcastable
BroadcastStyle(::Type{<:Value}) = Style{Value}()
BroadcastStyle(s::Style{Value}, ::BroadcastStyle) = s
broadcastable(x::Value) = x     # This is necessary, default is collect(x) which loses Value
broadcasted(::Style{Value}, f, args...) = isrecording() ? f(Bcasted.(args)...).value : broadcasted(f, value.(args)...)

## Recording machinery
mutable struct Node
    Value::Value
    parents::Vector{Node}
    children::Vector{Node}
    outgrad
    cdr::Node
    Node(v::Value) = new(v, Node[], Node[], nothing) # leaves cdr unassigned
end

mutable struct Tape
    nodes::IdDict{Tracked,Node}
    head::Node
    tail::Node
    Tape() = new(IdDict{Value,Node}())  # leaves head/tail unassigned
end

# Fix iterate, first, last, cons!, collect, get/grad?
# Fix record => cons!
# Fix show
# Fix get
# Fix put

grad(t,x)=nothing
grad(t::Tape,x::Tracked)=(n=get(t.nodes,x,nothing); n===nothing ? n : n.outgrad)

import Base: iterate

function Base.iterate(t::Tape, s=nothing)
    if s == nothing
        if isempty(t.nodes)
            nothing
        else
            (t.head, t.head)
        end
    else

    end
end

# value(x::Tape)=first(x).Value.value

# const NIL = Param([])
# newtape() = (n=Node(NIL); n.cdr=n; Tape(NIL => n))
# Base.iterate(t::Tape,s=(t[NIL],t[NIL])) = ((p,n) = s; p = p.cdr; p === n ? nothing : (p, (p, n)))
Base.collect(t::Tape)=(a=Array{Node}(undef,length(t)-1); i=0; for n in t; a[i+=1]=n; end; a)


const _tapes = Tape[]

abstract type Arg{N} end

function differentiate(f, x...; o...)
    global _tapes
    duplicate(x)=(isa(x,Value) ? identity(x) : x)
    if !isempty(_tapes)       # PR#75: to avoid tape confusion
        x = map(duplicate,x)  # duplicate tracked function arguments.
        o = isempty(o) ? () : pairs(map(duplicate,values(o)))
    end
    tape = Tape()
    push!(_tapes, tape)
    result = nothing
    try
        result = f(x...; o...)
        if isa(result,Param); result = identity(result); end # fix #101.1: turn Param->Result
    catch e
        Base.show_backtrace(stdout, Base.catch_backtrace())
        pop!(_tapes); throw(e)
    end
    if pop!(_tapes) !== tape; error("Tape stack error"); end
    if !isa(result,Result); return result; end
    if !isa(value(result), Number); error("Only scalar valued functions supported."); end
    n1 = first(tape)
    if result !== n1.Value; error("Result not on tape"); end
    n1.outgrad = one(value(result))
    tm(r::Result,i::Int)=(r.func==broadcast ? "$(r.args[1]).[$(i-1)]" : "$(r.func)[$i]")
    for n in tape
        if n.outgrad == nothing; continue; end
        r = n.Value
        @inbounds for i in 1:length(n.parents)
            if !isassigned(n.parents, i); continue; end
            p = n.parents[i]
            @timer tm(r,i) (g = back(r.func, Arg{i}, n.outgrad, r, r.args...; r.kwargs...))
            @timer "sum_outgrads" (p.outgrad = sum_outgrads(p.outgrad, g))
        end
        if isempty(_tapes) && isa(r,Result) && n !== n1; gcnode(n); end  # save memory
    end
    return tape
end

default_gc(n::Node) = (n.outgrad=nothing; n.Value.value=nothing)
gcnode = default_gc
set_gc_function(f::Function) = (global gcnode = f)

# This allows argument expressions like @diff sin(sqrt(x)) which fail with differentiate
# because arguments get evaluated before the tape gets created.
macro diff(fx); :(differentiate(()->$(esc(fx)))); end

back(x...; o...) = throw(ArgumentError("AutoGrad does not yet support back"*string(typeof.(x)))) # fix #101.2: error instead of nothing

# Primitives with special tracked or bcasted args (Values) call forw:
function forw(f, args...; kwargs...)
    tm(f::Function,a::Tuple)=(f==broadcast ? "$(a[1])." : "$f")
    if any(i->isa(i,Bcasted), args)
        args = (f, args...)
        f = broadcast
    end
    argvals = value.(args)
    @timer tm(f,argvals) (result = f(argvals...; kwargs...))
    if isempty(_tapes); return result; end
    @timer "record" begin
        result = Result{typeof(result)}(result, f, args, kwargs)
        for tape in _tapes
            record(result, tape)
        end
    end
    return result
end

function record(r::Result,t::Tape)
    nargs = length(r.args)
    n = Node(r)
    n.parents = Array{Node}(undef, nargs)
    @inbounds for argnum = 1:nargs
        arg = r.args[argnum]
        if !isa(arg,Value); continue; end	
        p = cons!(arg, t)
        n.parents[argnum] = p
        push!(p.children, n)
    end
    cons!(n, t)
end

function cons!(v::Value, t::Tape)
    n = get(t, v, nothing)
    if n === nothing
        n = cons!(Node(v), t)
    end
    return n
end

function cons!(n::Node, t::Tape)
    m = t[NIL]
    if isempty(m.parents); push!(m.parents, n); end # used by last(tape)
    n.cdr = m.cdr
    m.cdr = t[n.Value] = n
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
