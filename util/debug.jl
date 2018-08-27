# This file is deprecated.

using Printf
using Base.Broadcast: Broadcasted
using AutoGrad: Node, Rec, Tape, UngetIndex, _tapes, Result, Param
import AutoGrad: _dbg, dumptape

# Uncomment the following line and include("../util/debug.jl") for debugging:
# macro dbg(x); esc(:((DBG && println(join(_dbg.($x),' '))))); end; DBG=false; dbg(x)=(global DBG=x); _dbg(x)=summary(x)
macro dbg(x); end

# To perform profiling of AutoGrad internals, uncomment the following line. Make sure to Pkg.add("TimerOutputs").
# using TimerOutputs; macro prof(label,ex); :(@timeit $(esc(label)) $(esc(ex))); end
macro prof(label,ex); esc(ex); end

# For disambiguating objects:
_objdict = Dict{String,Array{UInt64,1}}()

function _dbg(x)
    str = _dbg1(x)
    oid = objectid(x)
    oids = get!(_objdict, str, [oid])
    idx = findeq(oids,oid)
    if idx === 0; push!(oids,oid); idx=length(oids); end
    if idx === 1; return str; end
    return "$str#$idx"
end

# Pretty print for debugging:
_dbg1(x)=summary(x) # extend to define short printable representations
_dbg1(x::Tuple)=@sprintf("Tuple%s", _dbg.(x))
_dbg1(x::Node)=@sprintf("Node(%s)",_dbg(x.rec.value))
_dbg1(x::Rec)=@sprintf("Rec(%s)",_dbg(x.value))
_dbg1(x::Param)=@sprintf("P(%s)",_dbg(x.value))
_dbg1(x::Result)=@sprintf("R(%s)",_dbg(x.value))
_dbg1(x::Tape)="T$(findeq(_tapes,x))"
_dbg1(x::AbstractArray{T}) where {T} = length(x) < 10 ? string(x) : @sprintf("Array{%s}%s",_tstr(T),size(x))
_dbg1(x::AbstractArray{T}) where {T<:Node} = "T" # @sprintf("Array{%s}%s",_tstr(T),size(x))
_dbg1(x::AbstractDict{T,S}) where {T,S} = "Dict{$T,$S}($(length(x)))"
_dbg1(x::Real)=@sprintf("%.2g",x)
_dbg1(x::Symbol)=string(x)
_dbg1(x::String)=x
_dbg1(x::Char)=string(x)
_dbg1(x::Function)=(s=replace(string(x),r".*\."=>"");length(s)<20 ? s : "Function")
_dbg1(x::Broadcasted)=@sprintf("Broadcasted(%s)",_dbg(copy(x)))
_dbg1(x::UngetIndex)=@sprintf("UngetIndex(%s→%s[%s])",_dbg(x.value),_dbg(x.container),x.index)
_tstr(::Type{T}) where T = replace(replace(string(T),r".*\." => ""),r"\{.*" => "") # short types

import Base.show
show(io::IO, n::Rec) = print(io, _dbg(n))
show(io::IO, n::Node) = print(io, _dbg(n))
show(io::IO, n::Tape) = print(io, _dbg(n))
show(io::IO, n::UngetIndex)= print(io, _dbg(n))

Base.collect(t::Tape)=(a=Node[]; for n in t; push!(a,n); end; a)

function dumptapes(tps=_tapes)
    if isempty(tps); return; end
    for n in reverse(collect(tps[1]))
        r = n.rec
        s = isa(r,Result) ? @sprintf("%s=%s%s",_dbg(r),_dbg(r.func),r.args) : _dbg(r)
        @printf("%-24s", s)
        for t in tps
            g = get(t,r,nothing)
            c = (g===nothing ? "-" : g.outgrad===nothing ? "0" : _dbg(g.outgrad))
            print("\t$c")
        end
        println()
    end
    println()
end

function dumptape(t::Tape)
    og(r) = map(t->(_dbg(t)*"="*_dbg(findgrad(t,r))), AutoGrad._tapes)
    println(_dbg(t))
    for i = 1:length(t)
        n = t[i]
        r = n.rec
        if isa(r,Param)
            v = _dbg(r.value)
            println(join(("$i. Param($v)", og(r)...), ' '))
        else
            p = ntuple(length(n.parents)) do j
                isassigned(n.parents,j) ? findeq(t,n.parents[j]) : 0
            end
            f = _dbg(r.func)
            println(join(("$i. $f$p", og(r)...), ' '))
        end
    end
    println()
end

findgrad(t,r)=(n=AutoGrad.findnode(t,r); n==nothing ? missing : n.outgrad)

function _debugtape(result, argvals)
    if !isa(result,Rec); return; end
    @assert length(result.tapes) == length(result.nodes) == 1
    tp = result.tapes[1]
    n = result.nodes[1]
    i = findeq(tp,n)
    p = ntuple(length(n.parents)) do j
        if isassigned(n.parents,j)
            findeq(tp,n.parents[j])
        elseif isa(argvals[j],Number) || isa(argvals[j],Symbol) || isa(argvals[j],AbstractRange)
            argvals[j]
        else
            0
        end
    end
    println("$i. $f$p")
end

# findfirst uses == which is inefficient for tapes, so we define findeq with ===
function findeq(A,v)
    @inbounds for i=1:length(A)
        if A[i] === v
            return i
        end
    end
    return 0
end

