using Printf
using Base.Broadcast: Broadcasted
using AutoGrad: Node, Rec, Tape, UngetIndex, findeq
import AutoGrad: _dbg

# For disambiguating objects:
_objdict = Dict{String,Array{UInt64,1}}()

function _dbg(x)
    str = _dbg1(x)
    oid = objectid(x)
    oids = get!(_objdict, str, [oid])
    idx = findeq(oids,oid)
    if idx === 0; push!(oids,oid); idx=length(oids); end
    if idx === 1; return str; end
    return "$str.$idx"
end

# Pretty print for debugging:
_dbg1(x)=summary(x) # extend to define short printable representations
_dbg1(x::Tuple)=@sprintf("Tuple%s", _dbg.(x))
_dbg1(x::Node)=@sprintf("Node(%s)",_dbg(x.rec.value))
_dbg1(x::Rec)=(n=length(x.nodes); @sprintf("Rec%s(%s)",n==1 ? "" : "[$n]", _dbg(x.value)))
_dbg1(x::AbstractArray{T}) where {T} = length(x) < 10 ? string(x) : @sprintf("Array{%s}%s",_tstr(T),size(x))
_dbg1(x::AbstractArray{T}) where {T<:Node} = @sprintf("Array{%s}%s",_tstr(T),size(x))
_dbg1(x::AbstractDict{T,S}) where {T,S} = "Dict{$T,$S}($(length(x)))"
_dbg1(x::Real)=@sprintf("%.2g",x)
_dbg1(x::Symbol)=string(x)
_dbg1(x::String)=x
_dbg1(x::Char)=string(x)
_dbg1(x::Function)=(s=string(x);length(s)<20 ? s : "Function")
_dbg1(x::Broadcasted)=@sprintf("Broadcasted(%s)",_dbg(copy(x)))
_dbg1(x::UngetIndex)=@sprintf("UngetIndex(%s→%s[%s])",_dbg(x.value),_dbg(x.container),x.index)
_tstr(::Type{T}) where T = replace(replace(string(T),r".*\." => ""),r"\{.*" => "") # short types

import Base.show
show(io::IO, n::Rec) = print(io, _dbg(n))
show(io::IO, n::Node) = print(io, _dbg(n))
show(io::IO, n::Tape) = print(io, _dbg(n))
show(io::IO, n::UngetIndex)= print(io, _dbg(n))

function dumptape(t::Tape)
    for i = 1:length(t)
        n = t[i]
        r = n.rec
        p = ntuple(length(n.parents)) do j
            isassigned(n.parents,j) ? findfirst(t,n.parents[j]) : 0
        end
        f = r.func
        println("$i. $f$p")
    end
end


