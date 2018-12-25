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

mutable struct Bcasted{T} <: Value{T}
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

import .Broadcast: BroadcastStyle, Style, broadcastable, broadcasted
BroadcastStyle(::Type{<:Value}) = Style{Value}()
BroadcastStyle(s::Style{Value}, ::BroadcastStyle) = s
broadcastable(x::Value) = x     # This is necessary, default is collect(x) which loses Value
Bcasted(v::T) where {T} = Bcasted{T}(v)
broadcasted(::Style{Value}, f, args...) = recording() ? f(Bcasted.(args)...).value : broadcasted(f, value.(args)...)


## Recording: primitive fns with Value args call forw

const _tapes = Tape[]
recording() = !isempty(_tapes)

forw(f, args...; kwargs...) = recording() ? bcast(f, args, kwargs) : fforw(f, args, kwargs)

function fforw(f, args, kwargs)
    @timer "fforw" begin
        @assert !recording()
        aval = args
        @inbounds for i in 1:length(aval)
            if isa(aval[i], Value)
                if aval === args
                    aval = Any[args...]
                end
                # @assert !isa(aval[i], Result) # This can happen during back
                @assert !isa(aval[i], Bcasted)
                aval[i] = aval[i].value
                @assert !isa(aval[i], Value) "Illegal value recursion: $(typeof(args[i]))"
            end
        end
        @assert aval !== args "forw called without Value args"
    end
    f(aval...; kwargs...)
end

function bcast(f, args, kwargs)
    @timer "bcast" begin
        @assert recording()
        aval = args
        @inbounds for i in 1:length(aval)
            if isa(aval[i], Bcasted)
                if aval === args
                    aval = Any[args...]
                end
                aval[i] = aval[i].value
                @assert !isa(aval[i], Bcasted)
            end
        end
        bcasted = (aval !== args)
        if bcasted && f !== broadcast
            aval = pushfirst!(aval, f)
            f = broadcast
        end
    end
    v = track(f, aval, kwargs, bcasted)
    bcasted ? Bcasted(v) : v
end

function track(f, args, kwargs, bcasted)
    @timer "track" begin
        @assert recording()
        aval = args
        @inbounds for i in 1:length(aval)
            if isa(aval[i], Tracked)
                if aval === args
                    aval = isa(args, Array) ? copy(args) : Any[args...]
                end
                aval[i] = aval[i].value
                @assert !isa(aval[i], Value)
            end
        end
    end
    if aval === args
        @assert bcasted "Tracking function without Value args."
        f(args...; kwargs...)
    else
        @timer ftimer(f,aval) (v = f(aval...; kwargs...))
        @timer "record" Result(v, f, args, kwargs)
    end
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
    if pop!(_tapes) !== tape; error("Tape stack error"); end
    if !isa(result,Result); return result; end
    if !isa(result.value, Number); error("Only scalar valued functions supported."); end
    n1 = first(tape.list)
    if result !== n1.Value; error("Result not on tape"); end
    n1.outgrad = one(result.value)
    for n in tape.list
        if n.outgrad == nothing; continue; end
        r = n.Value
        @inbounds for i in 1:length(n.parents)
            if !isassigned(n.parents, i); continue; end
            p = n.parents[i]
            @timer btimer(r,i) (g = back(r.func, Arg{i}, n.outgrad, r, r.args...; r.kwargs...))
            @timer "sum_outgrads" (p.outgrad = sum_outgrads(p.outgrad, g))
        end
        if isempty(_tapes) && isa(r,Result) && n !== n1; gcnode(n); end  # save memory
    end
    return tape
end

# back is defined by the @primitive macro
back(x...; o...) = throw(ArgumentError("AutoGrad does not yet support back"*string(typeof.(x)))) # fix #101.2: error instead of nothing

# Used by @timer
btimer(r::Result,i::Int)=(r.func===broadcast ? "$(r.args[1]).[$(i-1)]" : "$(r.func)[$i]")
ftimer(f::Function,a::Array{Any})=(f===broadcast ? "$(a[1])." : "$f")


## Exported functions:

Param(v::T,o=nothing) where {T} = Param{T}(v,o)

# This allows argument expressions like @diff sin(sqrt(x)) which fail with differentiate
# because arguments get evaluated before the tape gets created.
macro diff(fx); :(differentiate(()->$(esc(fx)))); end

# value() should give a regular (non-Value) result regardless of recursion
value(x) = x
value(x::Value) = x.value
value(x::Value{<:Value}) = error("Illegal type recursion $(typeof(x))")
value(x::Bcasted{<:Tracked}) = value(x.value) # Only type of Value recursion allowed
value(t::Tape)=first(t.list).Value.value

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
