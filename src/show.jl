# https://docs.julialang.org/en/stable/manual/types/#man-custom-pretty-printing-1
# Base.show(io::IO, z): single line format used in show, print, inside other objects.
# Base.show(io::IO, ::MIME"text/plain", z): multi-line format used by display.
# Base.show(io::IO, ::MIME"text/html", z): multi-line format for html output.
# get(io, :compact, false), show(IOContext(stdout, :compact=>true),z) for compact (array) printing.
# summary(io::IO, x) = print(io, typeof(x))
# string(z): uses print_to_string.

import Base: show

# One line show used for show, print, string etc.
show(io::IO, x::Param)  = print(IOContext(io,:compact=>true), "P(", value(x), ")")
show(io::IO, x::Result) = print(IOContext(io,:compact=>true), "R(", value(x), ")")
show(io::IO, x::Tape)   = print(IOContext(io,:compact=>true), "T(", value(x), ")")

# Multi line show used for display:
show(io::IO, ::MIME"text/plain", x::Tape) = show(io, x)

# Hack to take advantage of array display:
import Base: summary, size, getindex
struct ArrayValue{T,N} <: AbstractArray{T,N}; p; end
show(io::IO, m::MIME"text/plain", x::Value{A}) where {A<:AbstractArray} = show(io, m, ArrayValue{eltype(x),ndims(x)}(x))
size(p::ArrayValue) = size(p.p)
getindex(p::ArrayValue,i...) = getindex(p.p,i...)
summary(io::IO, x::Value{A}) where {A<:AbstractArray} = print(io, Base.dims2string(size(x)), " ", typeof(x))
summary(io::IO, p::ArrayValue) = summary(io, p.p)

function show(io::IO, n::Node)
    og(n::Node)=(n.outgrad === nothing ? 0 : n.outgrad)
    io = IOContext(io,:compact=>true)
    if isa(n.Value, Param)
        print(io, "N(", og(n), ", ", n.Value, ")")
    else
        r = n.Value
        print(io, "N(", og(n), ", ", r, " = ", r.func, (r.args..., r.kwargs...))
    end
end

function show(io::IO, ::MIME"text/plain", ts::Vector{Tape}) # to dump _tapes
    if isempty(ts); show(io, ts); end
    og(t::Tape,r::Value)=(n=get(t,r,nothing); n===nothing ? '-' : n.outgrad===nothing ? '0' : n.outgrad)
    io = IOContext(io,:compact=>true)
    for n in reverse(collect(ts[1]))
        r = n.Value
        for t in ts
            print(io, og(t,r), "\t")
        end
        if isa(r,Result)
            println(io, r, " = ", r.func, (r.args..., r.kwargs...))
        else
            println(io, r)
        end
    end
end
