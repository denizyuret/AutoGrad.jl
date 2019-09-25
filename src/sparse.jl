import Base: sum, zero, ones, length, size

struct Sparse
    container
    values
    indices
end

# Do we need these?
# sum(b::Sparse)=sum(sum(v) for v in b.values)
# zero(b::Sparse)=Sparse(b.container,[],[])
# ones(b::Sparse)=ones(b.container)
# length(b::Sparse)=length(b.container)
# size(b::Sparse,d...)=size(b.container,d...)

full(x)=x
full(b::Sparse)=sum_outgrads(zeroslike(b.container), b) # Try to avoid full() to conserve memory
zeroslike(a::AbstractArray{T}) where T = (isbitstype(T) ? zero(a) : Array{Any}(nothing,size(a)))

# We do not create Sparse for these types any more:
# zeroslike(a::AbstractDict)=empty(a)
# zeroslike(a::Tuple)=ntuple(i->nothing, length(a))
# zeroslike(a::Sparse)=zeroslike(a.container)
# zeroslike(a::T) where {T<:Number} = T(0)   # This comes up if people use getindex on a single number

import Base: *, +, -, /
import Base.Broadcast: broadcasted
for f in (:+, :-, :*, :/); @eval begin
    $f(s::Sparse, n::Number)=Sparse(s.container, [$f(v, n) for v in s.values], s.indices)
    $f(n::Number, s::Sparse)=Sparse(s.container, [$f(n, v) for v in s.values], s.indices)
    broadcasted(::typeof($f), s::Sparse, n::Number)=Sparse(s.container, [broadcast($f,v,n) for v in s.values], s.indices)
    broadcasted(::typeof($f), n::Number, s::Sparse)=Sparse(s.container, [broadcast($f,n,v) for v in s.values], s.indices)
end; end

+(a::AbstractArray, s::Sparse) = sum_outgrads(a, s)
+(s::Sparse, a::AbstractArray) = sum_outgrads(a, s)
-(s::Sparse) = -1*s
-(a::AbstractArray, s::Sparse) = sum_outgrads(a, -s)
-(s::Sparse, a::AbstractArray) = sum_outgrads(-a, s)
