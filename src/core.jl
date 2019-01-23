## Types:

abstract type Value{T} end

abstract type Tracked{T} <: Value{T} end

mutable struct Param{T} <: Tracked{T}
    value::T
    opt
    Param{T}(v,o) where {T} = new(v,o)
    Param{T}(v,o) where {T<:Value} = error("Param cannot take $T as arg.")
end

mutable struct Result{T} <: Tracked{T}
    value::Union{T,Nothing}     # gcnode sets this to nothing to save memory
    func
    args
    kwargs
    Result{T}(v,f,a,k) where {T} = new(v,f,a,k)
    Result{T}(v,f,a,k) where {T<:Value} = error("Result cannot take $T as arg.")
end

struct Bcasted{T} <: Value{T}
    value::T
    Bcasted{T}(v) where {T} = new(v)     # Bcasted{Tracked} is the only Value{Value} allowed
    Bcasted{T}(v) where {T<:Bcasted} = v # We do not want Bcasted{Bcasted}
end

mutable struct Node
    Value::Tracked
    parents::Vector{Node}
    children::Vector{Node}
    outgrad
    Node(v::Tracked) = new(v, Node[], Node[], nothing)
end

mutable struct Tape
    dict::IdDict{Tracked,Node}
    list::Vector{Node}
    Tape() = new(IdDict{Tracked,Node}(), Vector{Node}())
end

abstract type Arg{N} end


## Broadcasting: non-primitive fns broadcasted over Value args call themselves with Bcasted args

import .Broadcast: BroadcastStyle, broadcastable, broadcasted
using .Broadcast: Style, Broadcasted
BroadcastStyle(::Type{<:Value}) = Style{Value}()
BroadcastStyle(s::Style{Value}, ::BroadcastStyle) = s
broadcastable(x::Value) = x     # This is necessary, default is collect(x) which loses Value
Bcasted(v::T) where {T} = Bcasted{T}(v)
broadcasted(::Style{Value}, f, args...) = recording() ? fvalue(f(Bcasted.(args)...)) : broadcasted(f, value.(args)...)
Base.copyto!(x::Value,y) = copyto!(fvalue(x),y) # This is used by p .-= g when p is Param.


## Recording: primitive fns with Value args call forw

const _tapes = Tape[]
recording() = !isempty(_tapes)

# forw() is called with primitive functions that have Tracked or Bcasted args
function forw(f, args...; kwargs...)
    @timer "forwargs"        ((f, nobcast, novalue) = forwargs(f, args))
    @timer ftimer(f,novalue) (v = f(novalue...; kwargs...))
    if recording()
        if v isa Broadcasted
            @timer "unfuse"  (v = copy(v))
        end
        if novalue !== nobcast  # there are tracked args
            @timer "record"  (v = Result(v, f, nobcast, kwargs))
        end
        if nobcast !== args     # there are bcasted args
            @timer "bcasted" (v = Bcasted(v))
        end
    end
    return v
end

# Return two arg lists, one stripped of Bcasted and one stripped of all Values.
# Do not allocate unless necessary.
function forwargs(f, args)
    nobcast = novalue = args
    @inbounds for i in 1:length(args)
        if isa(nobcast[i], Bcasted)
            if nobcast === args; nobcast = Any[args...]; end
            if novalue === args; novalue = nobcast; end
            nobcast[i] = fvalue(nobcast[i])
            @assert !isa(nobcast[i], Bcasted) "Illegal value recursion: $(typeof(args[i]))"
        end
        if isa(novalue[i], Value)
            if novalue === args; novalue = Any[args...]
            elseif novalue === nobcast; novalue = copy(nobcast); end
            novalue[i] = value(novalue[i])
        end
    end
    @assert novalue !== args "forw called without Value args"
    if nobcast !== args
        @assert recording() "Bcasted args out of recording context"
        if f !== broadcasted
            pushfirst!(nobcast, f)
            if novalue !== nobcast; pushfirst!(novalue, f); end
            f = broadcasted
        end
    end
    return (f, nobcast, novalue)
end

function Result(v::T, f, args, kwargs) where {T}
    record!(t::Tape, v::Tracked) = (n = get(t.dict, v, nothing); n === nothing ? record!(t, Node(v)) : n)
    record!(t::Tape, n::Node) = (t.dict[n.Value] = n; pushfirst!(t.list, n); n)
    result = Result{T}(v, f, args, kwargs)
    narg = length(args)
    for tape in _tapes
        node = Node(result)
        node.parents = Array{Node}(undef, narg)
        @inbounds for i = 1:narg
            if isa(args[i], Tracked)
                node.parents[i] = record!(tape, args[i])
                push!(node.parents[i].children, node)
            end
        end
        record!(tape, node)
    end
    return result
end

Result(v::T, f, args, kwargs) where {T<:Tracked} = v  # Issue #106: no need to record twice


## Differentiate: call f recording primitives on tape, then call back on each primitive

function differentiate(f, x...; o...)
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
        Base.show_backtrace(stdout, Base.catch_backtrace()); println()
        pop!(_tapes); throw(e)
    end
    tape1 = pop!(_tapes)
    @assert tape1 === tape "Tape stack error"
    if !isa(result,Result); return result; end
    @assert isa(fvalue(result), Number) "Only scalar valued functions supported."
    resultnode = findresult(tape, result)
    resultnode.outgrad = one(fvalue(result))
    for n in tape.list
        if n.outgrad == nothing; continue; end
        r = n.Value
        @inbounds for i in 1:length(n.parents)
            if !isassigned(n.parents, i); continue; end
            p = n.parents[i]
            @timer btimer(r,i) (g = back(getfield(r, :func), Arg{i}, n.outgrad, r, getfield(r, :args)...; getfield(r, :kwargs)...))
            @timer "sum_outgrads" (p.outgrad = sum_outgrads(p.outgrad, g))
        end
        if isempty(_tapes) && isa(r,Result) && n !== resultnode; gcnode(n); end  # save memory
    end
    return tape
end

# back is defined by the @primitive macro
back(x...; o...) = throw(MethodError(back,x)) # fix #101.2: error instead of nothing

# Used by @timer
btimer(r::Result,i::Int)=(getfield(r, :func)===broadcasted ? "$(getfield(rm, :args)[1]).[$(i-1)]" : "$(getfield(r, :func))[$i]")
ftimer(f::Function,a::Array{Any})=(f===broadcasted ? "$(a[1])." : "$f")

# 99% result will be on tape.list[1] (last thing recorded), this handles the other 1% where
# the loss fn computes stuff recorded on tape after result but returns result at the end.
function findresult(tape::Tape, result::Result)
    if tape.list[1].Value === result; return tape.list[1]; end
    @inbounds for i in 2:length(tape.list)
        if tape.list[i].Value === result
            tape.list = tape.list[i:end]
            break
        end
    end
    @assert tape.list[1].Value === result "result not found on tape"
    return tape.list[1]
end

## Exported functions:

Param(v::T,o=nothing) where {T} = Param{T}(v,o)

# This allows argument expressions like @diff sin(sqrt(x)) which fail with differentiate
# because arguments get evaluated before the tape gets created.
macro diff(fx); :(differentiate(()->$(esc(fx)))); end

# value() should give a regular (non-Value) result regardless of recursion
value(x) = x
value(x::Value) = fvalue(x)
value(x::Value{<:Value}) = error("Illegal type recursion $(typeof(x))")
value(x::Bcasted{<:Tracked}) = value(fvalue(x)) # Only type of Value recursion allowed
value(t::Tape)=(isempty(t.list) ? nothing : fvalue(first(t.list).Value))

# on the contrary, fvalue() just get the direct field
fvalue(x) = getfield(x, :value)

# New style grad
grad(t,x)=nothing
grad(t::Tape,x::Tracked)=(n=get(t.dict,x,nothing); n===nothing ? n : n.outgrad)

# Old style grad and gradloss
function grad(fun::Function, argnum::Int=1, loss=false)
    function gradfun(args...; kwargs...)
        arg_wrt = args[argnum]
        if !isa(arg_wrt,Value); arg_wrt = Param(arg_wrt); end
        args = Any[args...]
        args[argnum] = arg_wrt
        result = differentiate(fun, args...; kwargs...)
        xgrad = isa(result, Tape) ? last(result.list).outgrad : nothing
        return loss ? (xgrad,value(result)) : xgrad
    end
    return gradfun
end

gradloss(f,a=1)=grad(f,a,true)

# Override gcnode for memory cleanup during back pass
default_gc(n::Node) = nothing # (n.outgrad=nothing; n.Value.value=nothing)
gcnode = default_gc
set_gc_function(f::Function) = (global gcnode = f)
