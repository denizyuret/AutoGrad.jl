"""
    Sparse(container, values, indices)

Create a sparse container to make gradient calculations efficient. `s::Sparse` represents
the value `a` as defined below:

    a = zero(s.container)
    for (idx, val) in zip(s.indices, s.values)
        a[idx] .+= val
    end

except when there are repeated indices in `idx`, the corresponding values get added rather
than being overwritten. See https://github.com/JuliaLang/julia/issues/31392.
"""
struct Sparse{T,N} <: AbstractArray{T,N}
    container
    values
    indices
end

Sparse(a::AbstractArray{T,N},v,i) where {T,N} = Sparse{T,N}(a,v,i)

# To add a Sparse to an Array without allocating extra space, we need to use:
# a .+= s  OR  a .= a .+ s
# Both of which translate to:
# materialize!(a, broadcasted(+, a, s))

import Base: size, copyto!
using Base.Broadcast: Broadcasted

# This is used in broadcasted
size(b::Sparse,d...)=size(b.container,d...)

# This is used in materialize!
function copyto!(a::AbstractArray, bc::Broadcasted{S,A,F,X}) where
    {S, A, F <: Union{typeof(+),typeof(-)}, X <: Tuple{Any,Sparse}}
    (b,c) = bc.args
    if !(size(a) == size(b) == size(c.container))
        a .= bc.f.(b, full(c))
        return a
    end
    a === b || copyto!(a, b)
    F <: typeof(-) && (c = -c)
    addto!(a, c)
    return a
end

# These are used in Knet/src/update.jl:
import LinearAlgebra: axpy!, norm, lmul!
axpy!(a::Number, x::Sparse, y::AbstractArray) = addto!(y, a*x)
lmul!(a::Number, x::Sparse{T,N}) where {T,N} = Sparse{T,N}(x.container, [ a*v for v in x.values ], x.indices)

# This does not give the correct result when there are repeated indices, but should be good enough for gclip
norm(x::Sparse) = sqrt(sum(abs2, norm(v) for v in x.values))

# Convert to regular array -- use this as last resort
full(b::Sparse)=addto!(zeroslike(b.container), b)
zeroslike(a::AbstractArray{T}) where T = (isbitstype(T) ? zero(a) : Array{Any}(nothing,size(a)))
full(x)=x


# Arithmetic with numbers
import Base: *, +, -, /
import Base.Broadcast: broadcasted
*(s::Sparse, n::Number) = Sparse(s.container, [ v*n for v in s.values ], s.indices)
*(n::Number, s::Sparse) = Sparse(s.container, [ v*n for v in s.values ], s.indices)
/(s::Sparse, n::Number) = Sparse(s.container, [ v/n for v in s.values ], s.indices)
broadcasted(::typeof(*), s::Sparse, n::Number) = Sparse(s.container, [ v.*n for v in s.values ], s.indices)
broadcasted(::typeof(*), n::Number, s::Sparse) = Sparse(s.container, [ v.*n for v in s.values ], s.indices)
broadcasted(::typeof(/), s::Sparse, n::Number) = Sparse(s.container, [ v./n for v in s.values ], s.indices)

# Arithmetic with arrays (can use addto! which overwrites its first argument)
+(a::AbstractArray, s::Sparse) = addto!(copy(a), s)
+(s::Sparse, a::AbstractArray) = addto!(copy(a), s)
-(a::AbstractArray, s::Sparse) = addto!(copy(a), -s)
-(s::Sparse, a::AbstractArray) = addto!(-a, s)
-(s::Sparse) = -1*s

# Issue #114: we may need to add multiple gradients
function +(a::Sparse, b::Sparse)
    @assert matches(a.container, b.container) "$(summary.((a.container, b.container)))"
    Sparse(a.container, [ a.values; b.values ], [ a.indices; b.indices ])
end

# Do we need these?
# sum(b::Sparse)=sum(sum(v) for v in b.values)
# zero(b::Sparse)=Sparse(b.container,[],[])
# ones(b::Sparse)=ones(b.container)
# length(b::Sparse)=length(b.container)

# We do not create Sparse for these types any more:
# zeroslike(a::AbstractDict)=empty(a)
# zeroslike(a::Tuple)=ntuple(i->nothing, length(a))
# zeroslike(a::Sparse)=zeroslike(a.container)
# zeroslike(a::T) where {T<:Number} = T(0)   # This comes up if people use getindex on a single number

