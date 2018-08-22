using Printf
using Base.Broadcast: Broadcasted
using AutoGrad: Node, Rec, Tape, UngetIndex

# Pretty print for debugging:
_dbg(x)=summary(x) # extend to define short printable representations
_dbg(x::Tuple)=join(_dbg.(x),' ')
_dbg(x::Node)=@sprintf("N%s(%s)",id2(x),_dbg(x.rec.value))
_dbg(x::Rec)=(n=length(x.nodes); @sprintf("R%s%s(%s)",id2(x),n==1 ? "" : "[$n]", _dbg(x.value)))
_dbg(x::Tape)="N"*id2(x)*ssize(x)
_dbg(x::AbstractArray)=_dbg(eltype(x))*id2(x)*ssize(x)
_dbg(::Type{Any})="A"
_dbg(::Type{Float32})="S"
_dbg(::Type{Float64})="D"
_dbg(t::Type)="$t"
_dbg(x::Dict)="H"*id2(x)
_dbg(x::Float32)=@sprintf("%.2g",x) # "S"*id2(x)
_dbg(x::Float64)=@sprintf("%.2g",x) # "D"*id2(x) 
_dbg(x::Symbol)=string(x)
_dbg(x::Integer)=string(x)
_dbg(x::String)=x
_dbg(x::Char)=x
_dbg(x::Function)=(s=string(x);length(s)<20 ? s : "F"*id2(x))
_dbg(x::Broadcasted)=@sprintf("B(%s)",_dbg(copy(x)))
_dbg(x::UngetIndex)="U$(id2(x))_$(_dbg(x.container))_$(_dbg(x.value))_$((x.index...,))"
id2(x)=@sprintf("%02d",(objectid(x)%100))
ssize(x)="$(collect(size(x)))"

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


