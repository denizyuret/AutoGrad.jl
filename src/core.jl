export Param, differentiate, df, gradient

abstract type Rec{T} end

mutable struct Param{T} <: Rec{T}
    value::T
end

mutable struct Result{T} <: Rec{T}
    value::T
    func::Function
    args::Tuple
    kwargs::Base.Iterators.Pairs
    Result(val::T, func, args...; kwargs...) where T = new{T}(val, func, args, kwargs)
end

mutable struct Node
    outgrad
    rec::Rec
    parents::Vector{Node}
    prev::Node
    Node(og,rc,pa,pr)=new(og,rc,pa,pr)
    Node()=new()
end

# Tape: rec0=>node0, rec1=>node1, ..., recN=>nodeN
# where node[n].prev = node[n-1]
# Special node0 marks both ends of the tape: 
# node1.prev = node0
# node0.prev = nodeN
# Special rec0 acts as a key to node0.
# mutable struct is a lot faster as key for an IdDict!
const Tape = IdDict{Rec,Node}
const EOT = Param([])
newtape() = (n=Node(); n.prev=n; Tape(EOT => n))
# Tape is iterated in reverse, last in first out
# This also makes first(tape) the final result node
Base.iterate(t::Tape,s=(t[EOT],t[EOT])) = 
    ((p,n) = s; p = p.prev; p === n ? nothing : (p, (p, n)))
# This hack is to make grad faster:
last(t::Tape)=t[EOT].parents[1]

gradient(t,x)=nothing
gradient(t::Tape,x::Rec)=(n=get(t,x,nothing); n===nothing ? n : n.outgrad)

getval(x)=x
getval(x::Rec)=x.value
getval(x::Tape)=first(x).rec.value

_tapes = Tape[]

function differentiate(f, x...; o...)
    global _tapes
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
    if !isa(result.value, Number); error("AutoGrad can only handle scalar valued functions"); end
    n = first(tape)
    if result !== n.rec; error("Result not on tape"); end
    n.outgrad = one(result.value)
    for n in tape
        if n.outgrad == nothing; continue; end
        r = n.rec
        @inbounds for i in 1:length(n.parents)
            if !isassigned(n.parents, i); continue; end
            p = n.parents[i]
            g = back(r.func, Val(i), n.outgrad, r, r.args...; r.kwargs...)
            p.outgrad = sum_outgrads(p.outgrad, g)
        end
        #if !isa(r,Param); n.outgrad = nothing; end
    end
    return tape
end

const df = differentiate

function forw(f, args...; kwargs...)
    argvals = getval.(args)
    result = f(argvals...; kwargs...)
    if isempty(_tapes); return result; end
    result = Result(result, f, args...; kwargs...)
    for tape in _tapes
        push!(tape, result)
    end
    return result
end

function Base.push!(t::Tape, r::Rec)
    n = get(t,r,nothing)
    if n !== nothing; return n; end
    if isa(r,Result)
        nargs = length(r.args)
        parents = Array{Node}(undef, nargs)
        @inbounds for argnum = 1:nargs
            arg = r.args[argnum]
            if !isa(arg,Rec); continue; end
            parent = push!(t,arg)
            parents[argnum] = parent
        end
    else
        parents = Node[]
    end
    m = t[EOT]
    prev = m.prev
    node = Node(nothing, r, parents, prev)
    if !isdefined(m,:parents); m.parents = [node]; end
    t[r] = m.prev = node
end

# old interface
function grad(fun::Function, argnum::Int=1)
    function gradfun(args...; kwargs...)
        arg_wrt = args[argnum]
        arg_wrt = !isa(arg_wrt,Rec) ? Param(arg_wrt) : identity(arg_wrt) # from PR#75
        args = Any[args...]
        args[argnum] = arg_wrt
        result = differentiate(fun, args...; kwargs...)
        isa(result, Tape) ? last(result).outgrad : nothing
    end
    return gradfun
end


sum_outgrads(a::Number, b::Number)=a+b
sum_outgrads(a::Tuple, b::Tuple)=tuple([sum_outgrads(x,y) for (x,y) in zip(a,b)]...)
sum_outgrads(a::AbstractDict, b::AbstractDict) = (z=similar(a); for d in (a,b), (k,v) in d; z[k]=sum_outgrads(v,get(z,k,nothing)); end; z)
# We could have Array{Array} and Array{Any} added:
sum_outgrads(a::AbstractArray{T},b::AbstractArray) where T = (if isbitstype(T); (a+b); else; T[sum_outgrads(x,y) for (x,y) in zip(a,b)]; end)
# sum_outgrads needs to be a primitive for higher order gradients:
sum_outgrads(a::Rec,b::Rec)=forw(sum_outgrads,a,b)
sum_outgrads(a::Rec,b)=forw(sum_outgrads,a,b)
sum_outgrads(a,b::Rec)=forw(sum_outgrads,a,b)
back(::typeof(sum_outgrads),::Val{N},dy,y,x1,x2) where N = dy
# we use `nothing` to indicate zero gradients
sum_outgrads(::Nothing,::Nothing)=nothing
sum_outgrads(a::Rec,::Nothing)=a   # to avoid ambiguity
sum_outgrads(::Nothing,a::Rec)=a   # to avoid ambiguity
sum_outgrads(a,::Nothing)=a
sum_outgrads(::Nothing,a)=a

