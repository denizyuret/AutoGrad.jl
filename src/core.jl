export Param, differentiate, df, gradient, gr

abstract type Rec{T} end

# mutable structs are a lot faster as keys for an IdDict!
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
    cdr::Node
    Node(og,rc,pa,pr)=new(og,rc,pa,pr)
    Node()=new()
end

# Tape: recN=>nodeN, ..., rec1=>node1, NIL=>node0
const Tape = IdDict{Rec,Node}
# Special node0 marks both ends of the tape: 
# node[n].cdr = node[n-1]
# node1.cdr = node0
# node0.cdr = nodeN
# Special rec NIL acts as a key to node0.
const NIL = Param([])
newtape() = (n=Node(); n.cdr=n; Tape(NIL => n))
# Tape is iterated in reverse, last in first out
Base.iterate(t::Tape,s=(t[NIL],t[NIL])) = 
    ((p,n) = s; p = p.cdr; p === n ? nothing : (p, (p, n)))
# This automatically makes first(tape) the final result node
# last(tape) is the initial parameter node, but default slow.
# This hack is to make old style grad faster:
last(t::Tape)=t[NIL].parents[1]

gradient(t,x)=nothing
gradient(t::Tape,x::Rec)=(n=get(t,x,nothing); n===nothing ? n : n.outgrad)
const gr = gradient

getval(x)=x
getval(x::Rec)=x.value
getval(x::Tape)=first(x).rec.value

_tapes = Tape[]

function differentiate(f, x...; o...)
    global _tapes
    if !isempty(_tapes)       # PR#75: to avoid tape confusion
        x = map(duplicate,x)  # duplicate tracked function arguments.
        o = isempty(o) ? () : pairs(map(duplicate,o.data))
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
        #TODO: if !isa(r,Param); n.outgrad = nothing; end  #This breaks higher order
    end
    return tape
end

duplicate(x)=(isa(x,Rec) ? identity(x) : x)

const df = differentiate

function forw(f, args...; kwargs...)
    argvals = getval.(args)
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
        if !isa(arg,Rec); continue; end	
        p = get(t,arg,nothing)
        if p === nothing
            p = cons!(arg,t)
        end
        parents[argnum] = p
    end
    cons!(r,t,parents)
end

function cons!(r::Rec,t::Tape,parents::Vector{Node}=Node[])
    m = t[NIL]
    n = Node(nothing, r, parents, m.cdr)
    if !isdefined(m,:parents); m.parents = [n]; end # hack to make last(tape) faster
    m.cdr = t[r] = n
end

# old interface
function grad(fun::Function, argnum::Int=1)
    function gradfun(args...; kwargs...)
        arg_wrt = args[argnum]
        if !isa(arg_wrt,Rec); arg_wrt = Param(arg_wrt); end
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

