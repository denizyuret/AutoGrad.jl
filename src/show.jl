# https://docs.julialang.org/en/stable/manual/types/#man-custom-pretty-printing-1
# Base.show(io::IO, z): single line format used in show, print, inside other objects.
# Base.show(io::IO, ::MIME"text/plain", z): multi-line format used by display.
# Base.show(io::IO, ::MIME"text/html", z): multi-line format for html output.
# get(io, :compact, false), show(IOContext(stdout, :compact=>true),z) for compact (array) printing.
# summary(io::IO, x) = print(io, typeof(x))
# string(z): uses print_to_string.

import Base: show

# Summary string
valstr(x)=_valstr(value(x))
_valstr(::Nothing)="nothing"
_valstr(x)=(hasmethod(size,(typeof(x),)) && !isempty(size(x)) ? "$(typeof(x))($(join(size(x),',')))" : string(x))
_valstr(t::Tuple)="($(join(_valstr.(t),',')))"

# One line show used for show, print, string etc.
show(io::IO, x::Bcasted)  = print(IOContext(io,:compact=>true), "B(", valstr(x), ")")
show(io::IO, x::Param)    = print(IOContext(io,:compact=>true), "P(", valstr(x), ")")
show(io::IO, x::Result)   = print(IOContext(io,:compact=>true), "R(", valstr(x), ")")
show(io::IO, x::Tape)     = print(IOContext(io,:compact=>true), "T(", valstr(x), ")")

# Multi line show used for display:
show(io::IO, ::MIME"text/plain", x::Tape) = show(io, x)

# Hack to take advantage of array display:
import Base: summary, size, getindex
struct ArrayValue{T,N} <: AbstractArray{T,N}; p; end
show(io::IO, m::MIME"text/plain", x::Value{A}) where {A<:AbstractArray} = show(io, m, ArrayValue{eltype(x),ndims(x)}(x))
size(p::ArrayValue) = size(p.p.value)
getindex(p::ArrayValue,i...) = getindex(p.p.value,i...)
summary(io::IO, x::Value{A}) where {A<:AbstractArray} = print(io, Base.dims2string(size(x)), " ", typeof(x))
summary(io::IO, p::ArrayValue) = summary(io, p.p)

function show(io::IO, n::Node)
    og(n::Node)=(n.outgrad === nothing ? 0 : valstr(n.outgrad))
    io = IOContext(io,:compact=>true)
    if isa(n.Value, Param)
        print(io, "N(", og(n), ", ", n.Value, ")")
    else
        r = n.Value
        print(io, "N(", og(n), ", ", r, " = ", r.func, "(", join(valstr.(r.args),", "), 
              isempty(r.kwargs) ? "" : "; "*join(["$(x[1])=$(valstr(x[2]))" for x in r.kwargs], ", "), "))")
    end
end

function show(io::IO, ::MIME"text/plain", ts::Vector{Tape}) # to dump _tapes
    if isempty(ts); show(io, ts); return; end
    og(t::Tape,r::Value)=(n=get(t.dict,r,nothing); n===nothing ? '-' : n.outgrad===nothing ? '0' : valstr(n.outgrad))
    argstr(x)=(n=findfirst(a->(a.Value===x),ts[1].list); n===nothing ? valstr(x) : "R$(length(ts[1].list)+1-n)")
    io = IOContext(io,:compact=>true)
    for (i,n) in enumerate(reverse(ts[1].list))
        r = n.Value
        if isa(r,Result)
            print(io, "$i. ", valstr(r), " = ", r.func, "(", join(argstr.(r.args),", "), 
                  isempty(r.kwargs) ? "" : "; "*join(["$(x[1])=$(valstr(x[2]))" for x in r.kwargs], ", "), "))")
        else
            print(io, "$i. ", r)
        end
        println(io, " ∇=", join([og(t,r) for t in ts], '|'))
    end
end
