abstract type Value{T} end

abstract type Tracked{T} <: Value{T} end

mutable struct Bcasted{T} <: Value{T}
    value::T
    Bcasted{T}(v) where {T} = new(v)
    Bcasted{T}(v) where {T<:Bcasted} = v # We do not want Bcasted{Bcasted}
end
Bcasted(v::T) where {T} = Bcasted{T}(v)

mutable struct Param{T} <: Tracked{T}
    value::T; opt
    Param{T}(v,o) where {T} = new(v,o)
    Param{T}(v,o) where {T<:Value} = error("Param cannot take $T as arg.")
end
Param(v::T,o=nothing) where {T} = Param{T}(v,o)

mutable struct Result{T} <: Tracked{T}
    value::Union{T,Nothing}     # gcnode sets this to nothing to save memory
    func
    args
    kwargs
    Result{T}(v,f,a,k) where {T} = new(v,f,a,k)
    Result{T}(v,f,a,k) where {T<:Value} = error("Result cannot take $T as arg.")
end

# value() should give a regular (non-Value) result regardless of recursion
value(x) = x
value(x::Value) = x.value
value(x::Value{<:Value}) = error("Illegal type recursion $(typeof(x))")
value(x::Bcasted{<:Tracked}) = value(x.value) # Only type of Value recursion allowed

## Recording machinery
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

value(t::Tape)=first(t.list).Value.value
grad(t::Tape,x::Tracked)=(n=get(t.dict,x,nothing); n===nothing ? n : n.outgrad)
grad(t,x)=nothing

const _tapes = Tape[]
recording() = !isempty(_tapes)

# To catch whenever one arg is a Value in broadcast expressions, we define a style:
import .Broadcast: BroadcastStyle, Style, broadcastable, broadcasted
BroadcastStyle(::Type{<:Value}) = Style{Value}()
BroadcastStyle(s::Style{Value}, ::BroadcastStyle) = s
broadcastable(x::Value) = x     # This is necessary, default is collect(x) which loses Value

# This is the first place where Bcasted type is introduced
# This should only catch non-primitive functions
broadcasted(::Style{Value}, f, args...) = recording() ? f(Bcasted.(args)...).value : broadcasted(f, value.(args)...)

function forw(f, args...; kwargs...)
    if recording()
        bcast(f, args, kwargs)
    else
        fforw(f, args, kwargs)
    end
end

function fforw(f, args, kwargs)
    @assert !recording()
    aval = args
    @inbounds for i in 1:length(aval)
        if isa(aval[i], Value)
            if aval === args
                aval = Any[args...]
            end
            @assert isa(aval[i], Param) "$(typeof(aval[i])) while not recording."
            aval[i] = aval[i].value
            @assert !isa(aval[i], Value) "Illegal value recursion: $(typeof(args[i]))"
        end
    end
    if aval === args
        error("forw called without Value args")
    else
        f(aval...; kwargs...)
    end
end

function bcast(f, args, kwargs)
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
    if aval === args
        track(f, aval, kwargs, false)
    else
        aval = pushfirst!(aval, f)
        track(broadcast, aval, kwargs, true) |> Bcasted
    end
end

function track(f, args, kwargs, bcasted)
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
    if aval === args
        @assert bcasted
        return f(args...; kwargs...)
    else
        v = f(aval...; kwargs...)
        return Result(v, f, args, kwargs)
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

back(x...; o...) = throw(ArgumentError("AutoGrad does not yet support back"*string(typeof.(x)))) # fix #101.2: error instead of nothing

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
        Base.show_backtrace(stdout, Base.catch_backtrace()); println()
        pop!(_tapes); throw(e)
    end
    if pop!(_tapes) !== tape; error("Tape stack error"); end
    if !isa(result,Result); return result; end
    if !isa(value(result), Number); error("Only scalar valued functions supported."); end
    n1 = first(tape.list)
    if result !== n1.Value; error("Result not on tape"); end
    n1.outgrad = one(value(result))
    tm(r::Result,i::Int)=(r.func==broadcast ? "$(r.args[1]).[$(i-1)]" : "$(r.func)[$i]")
    for n in tape.list
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


### DEPRECATED:

# # Fix iterate, first, last, cons!, collect, get/grad?
# # Fix record => cons!
# # Fix show
# # Fix get
# # Fix put
# # Define recording


# import Base: iterate

# function Base.iterate(t::Tape, s=nothing)
#     if s == nothing
#         if isempty(t.nodes)
#             nothing
#         else
#             (t.head, t.head)
#         end
#     else

#     end
# end

# # 

# # const NIL = Param([])
# # newtape() = (n=Node(NIL); n.cdr=n; Tape(NIL => n))
# # Base.iterate(t::Tape,s=(t[NIL],t[NIL])) = ((p,n) = s; p = p.cdr; p === n ? nothing : (p, (p, n)))
# Base.collect(t::Tape)=(a=Array{Node}(undef,length(t)-1); i=0; for n in t; a[i+=1]=n; end; a)

# # Value recursion illegal except Bcasted{<:Tracked}
# checktype(x) = true
# checktype(x::Value{<:Value}) = false
# checktype(x::Bcasted{<:Tracked}) = checktype(x.value)
# checktypes(args)=(all(checktype, args) || error("Bad type $(typeof.(args))"))

# function cons!(v::Value, t::Tape)
#     n = get(t, v, nothing)
#     if n === nothing
#         n = cons!(Node(v), t)
#     end
#     return n
# end

# function cons!(n::Node, t::Tape)
#     m = t[NIL]
#     if isempty(m.parents); push!(m.parents, n); end # used by last(tape)
#     n.cdr = m.cdr
#     m.cdr = t[n.Value] = n
# end

# Base.last(t::Tape)=t[NIL].parents[1] # cons! makes sure this works.

# # forw may be called when recording or not recording (e.g. with a Param)
# # Bcasted and Result args should only appear when recording, Param can exist any time.

# # 1 scan args for Bcasted and strip
# # 2 if Bcasted found and not recording it is an error!
# # 3 if found replace f with broadcast
# # 4 scan args for Tracked and strip
# # 5 if Result found and not recording it is an error!
# # 6 we need to process Bcasted even if not recording? No, that's illegal. If !recording() only Param allowed.

# # Primitives with special tracked or bcasted args, i.e. f(::Value) call forw:
# function forwargs(f, args)
#     arg1 = Any[args...]
#     arg2 = copy(arg1)
#     isrecording = recording()
#     bcasting = tracking = resultfound = false
#     @inbounds for i in 1:length(arg1)
#         if isa(arg1[i], Bcasted)
#             bcasting = true
#             arg1[i] = arg1[i].value
#             arg2[i] = arg1[i]
#         end
#         if isa(arg2[i], Tracked)
#             if isa(arg2[i], Result)
#                 resultfound = true
#             end
#             tracking = true
#             arg2[i] = arg2[i].value
#             if isa(arg2[i], Value)
#                 error("Illegal Value recursion: $(typeof(args[i]))")
#             end
#         end
#     end
#     if resultfound && !isrecording
#         error("Result found while not recording.")
#     end
#     if bcasting
#         isrecording || error("Bcasted found while not recording.")
#         pushfirst!(arg1, f)
#         pushfirst!(arg2, f)
#         f = broadcast
#     end
#     (f, arg1, arg2, bcasting, tracking)
# end

# function forw(f, args...; kwargs...)
#     @timer "forwargs" ((f, arg1, arg2, bcasting, tracking) = forwargs(f, args))
#     @timer ftimer(f,args) (result = f(arg2...; kwargs...))
#     if recording()
#         @timer "record" begin
#             result = Result(result, f, arg1, kwargs) # arg1 is not a tuple!
#             for tape in _tapes
#                 record(tape, result)
#             end
#         end
#     end
#     if bcasting
#         result = Bcasted(result)
#     end
#     return result
# end



# ftimer(f,a)=(f===broadcast ? "$(a[1])." : "$f") # used by @timer

# function record(t::Tape, r::Result)
#     nargs = length(r.args)
#     n = Node(r)
#     n.parents = Array{Node}(undef, nargs)
#     @inbounds for argnum = 1:nargs
#         arg = r.args[argnum]
#         if !isa(arg,Tracked); continue; end	
#         p = cons!(arg, t)
#         n.parents[argnum] = p
#         push!(p.children, n)
#     end
#     cons!(n, t)
# end
