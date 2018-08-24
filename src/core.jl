export Param, differentiate, df, gradient

abstract type Rec{T} end

struct Param{T} <: Rec{T}
    value::T
end

struct Result{T} <: Rec{T}
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
    Node(r::Result) = new(nothing, r, Array{Node}(undef, length(r.args)))
    Node(r::Param) = new(nothing, r, Array{Node}[])
end

const Tape = Vector{Node}

getval(x)=x
getval(x::Rec)=x.value
getval(x::Tape)=x[end].rec.value

global _tapes = []

function differentiate(f, x...; o...)
    global _tapes
    tape = Tape()
    push!(_tapes, tape)
    result = nothing
    try
        result = f(x...; o...)
    catch e
        pop!(_tapes)
        throw(e)
    end
    if pop!(_tapes) !== tape; error("Tape stack error"); end
    if !isa(result,Result); return result; end
    if !isa(result.value, Number); error("diff can only handle scalar valued functions"); end
    if result !== tape[end].rec; error("Result not on tape"); end

    tape[end].outgrad = one(result.value)
    for j = length(tape):-1:1
        n = tape[j]
        if n.outgrad == nothing; continue; end
        r = n.rec
        for i in 1:length(n.parents)
            if !isassigned(n.parents, i); continue; end
            p = n.parents[i]
            g = back(r.func, Val(i), n.outgrad, r, r.args...; r.kwargs...)
            p.outgrad = sum_outgrads(p.outgrad, g)
        end
        #if !isa(r,Param); tape[j] = nothing; end
    end
    return tape
end

const df = differentiate

function forw(f, args...; kwargs...)
    global _tapes
    argvals = getval.(args)
    result = f(argvals...; kwargs...)
    if isempty(_tapes); return result; end
    result = Result(result, f, args...; kwargs...)
    for tape in _tapes
        rnode = Node(result)
        for argnum = 1:length(args)
            arg = args[argnum]
            if !isa(arg,Rec); continue; end
            parent = findnode!(tape,arg)
            rnode.parents[argnum] = parent
        end
        push!(tape, rnode)
    end
    return result
end

function findnode(t::Tape,x::Rec)
    for n in t
        if n.rec === x
            return n
        end
    end
    return nothing
end

function findnode!(t::Tape,x::Rec)
    for n in t
        if n.rec === x
            return n
        end
    end
    #TODO: do we always push all nodes?
    node = Node(x)
    push!(t, node)
    return node
end

function gradient(t::Tape,x::Rec)
    n = findnode(t,x)
    n == nothing ? n : n.outgrad
end

gradient(t,x)=nothing

function grad(fun::Function, argnum::Int=1)
    function gradfun(args...; kwargs...)
        arg_wrt = args[argnum]
        arg_wrt = isa(arg_wrt,Rec) ? identity(arg_wrt) : Param(arg_wrt) # identity(arg_wrt) from PR#75
        args = Any[args...]
        args[argnum] = arg_wrt
        result = differentiate(fun, args...; kwargs...)
        isa(result, Tape) ? result[1].outgrad : nothing
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

