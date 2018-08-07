# Pretty print for debugging:
_dbg(x)=summary(x) # extend to define short printable representations
_dbg(x::Tuple)=string(map(_dbg,x)...)
_dbg(x::Node)=_dbg(x.rec.value)*"N"
_dbg(x::Rec)=_dbg(x.value)*"R"
_dbg(x::Tape)="N"*ssize(x)
_dbg(x::AbstractArray)=_dbg(eltype(x))*ssize(x)*id2(x)
_dbg(::Type{Any})="A"
_dbg(::Type{Float32})="S"
_dbg(::Type{Float64})="D"
_dbg(t::Type)="$t"
_dbg(x::Dict)="H"*id2(x)
_dbg(x::Float32)="S"*id2(x)
_dbg(x::Float64)="D"*id2(x)
_dbg(x::Symbol)="$x"
_dbg(x::Integer)="$x"
id2(x)="$(objectid(x)%1000)"
ssize(x)="$(collect(size(x)))"

Base.show(io::IO, n::Rec) = print(io, _dbg(n))
Base.show(io::IO, n::Node) = print(io, _dbg(n))
Base.show(io::IO, n::Tape) = print(io, _dbg(n))

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

