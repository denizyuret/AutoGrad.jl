# Use `perl ../deps/imports.pl base.jl` to generate the next line
import Base: !=, !==, *, +, -, /, <, <=, ==, >, >=, \, ^, abs, abs2, all, any, axes, big, broadcast, ceil, checkbounds, copy, count, div, eachindex, eltype, eps, float, floor, identity, isassigned, isempty, isequal, isfinite, isinf, isinteger, isless, isnan, lastindex, length, literal_pow, maximum, minimum, ndims, oftype, one, ones, permutedims, prod, rem, reshape, round, sign, signbit, similar, size, stride, strides, sum, trunc, typemax, typemin, unsafe_trunc, values, vec, widemul, zero
import Base.Broadcast: broadcasted

# The following list copied from relevant portions of julia/base/exports.jl

### Operators
# !   Bool function
@zerograd !=(x1,x2)
# ≠   Same as !=
@zerograd !==(x1,x2)
# ≡   Same as ===, Core builtin function, cannot add methods
# ≢   Same as !==
# xor Int function
# ⊻   Same as xor
# %   Same as rem
# ÷   Same as div
# &   Int function

@primitive +(x),dy  dy
@primitive +(x1,x2),dy  unbroadcast(x1,dy)  unbroadcast(x2,dy)
@primitive -(x),dy  -dy
@primitive -(x1,x2),dy  unbroadcast(x1,dy)  unbroadcast(x2,-dy)

@primitive *(x),dy                      dy
@primitive1 *(x1,x2),dy                 (dy*x2')  (x1'*dy)
@primitive2 *(x1,x2),dy                 unbroadcast(x1,dy.*x2)  unbroadcast(x2,x1.*dy)
@primitive *(x1::Number,x2::Number),dy  unbroadcast(x1,dy.*x2)  unbroadcast(x2,x1.*dy)
@primitive *(x1::Number,x2),dy          unbroadcast(x1,dy.*x2)  unbroadcast(x2,x1.*dy)
@primitive *(x1,x2::Number),dy          unbroadcast(x1,dy.*x2)  unbroadcast(x2,x1.*dy)

@primitive1 /(x1,x2),dy                 error("A/B grad undefined") error("A/B grad undefined") #TODO
@primitive2 /(x1,x2),dy                 unbroadcast(x1,dy./x2)  unbroadcast(x2,-dy.*x1./abs2.(x2))
@primitive /(x1::Number,x2::Number),dy  unbroadcast(x1,dy./x2)  unbroadcast(x2,-dy.*x1./abs2.(x2))
@primitive /(x1::Number,x2),dy          unbroadcast(x1,dy./x2)  unbroadcast(x2,-dy.*x1./abs2.(x2))
@primitive /(x1,x2::Number),dy          unbroadcast(x1,dy./x2)  unbroadcast(x2,-dy.*x1./abs2.(x2))

@primitive1 \(x1,x2),dy                 error("A\\B grad undefined") error("A\\B grad undefined") #TODO
@primitive2 \(x1,x2),dy                 unbroadcast(x1,-dy.*x2./abs2.(x1))  unbroadcast(x2,dy./x1)
@primitive \(x1::Number,x2::Number),dy  unbroadcast(x1,-dy.*x2./abs2.(x1))  unbroadcast(x2,dy./x1)
@primitive \(x1::Number,x2),dy          unbroadcast(x1,-dy.*x2./abs2.(x1))  unbroadcast(x2,dy./x1)
@primitive \(x1,x2::Number),dy          unbroadcast(x1,-dy.*x2./abs2.(x1))  unbroadcast(x2,dy./x1)

@primitive1 ^(x1,x2),dy,y                  error("A^B grad undefined") error("A^B grad undefined") #TODO: Arr^Int (Arr^Num can be imaginary)
@primitive2 ^(x1,x2),dy,y                  unbroadcast(x1,dxndx(x1,x2,dy))  unbroadcast(x2,dy.*y.*log.(x1))
@primitive ^(x1::Number,x2::Number),dy,y   dxndx(x1,x2,dy)  dy*y*log(x1)
literal_pow(::typeof(^), x::Value, ::Val{N}) where N = forw(^,x,N) # x^p for any literal integer p is lowered to Base.literal_pow(^, x, Val(p))
broadcasted(::typeof(literal_pow), ::typeof(^), x::Value, ::Val{N}) where N = forw(broadcasted,^,x,N)
dxndx(x1,x2,dy)=(if x2==0; zero(dy); elseif x2==1; dy; elseif x2==2; 2 .* x1 .* dy; else; dy.*x2.*x1.^(x2 .- 1); end) # optimize common cases

# //  Int function
@zerograd <(x1,x2)
# <:  Type function
# <<  Int function
@zerograd <=(x1,x2)
# ≤   Same as <=
@zerograd ==(x1,x2)
@zerograd >(x1,x2)
# >:  Type function
@zerograd >=(x1,x2)
# ≥   Same as >=
# >>  Int function
# >>> Int function

# |   Int function
# |>  Function chaining
# ~   Int function
# :   Range operator
# =>  Pair constructor
# ∘   Function composition

### scalar math (some of this is in math.jl)
@primitive abs(x),dy (dy.*sign.(x))
@primitive abs2(x),dy (dy.*(2x))
# angle(z): Compute the phase angle in radians of a complex number z.
@primitive big(x),dy oftype(x,dy)
# binomial(n,k): Number of ways to choose k out of n items.
# bswap(n): Reverse the byte order of n.
@zerograd ceil(x)
# cis(z): Return \exp(iz).
# cld(x, y): Smallest integer larger than or equal to x/y.
# cmp(x,y): Return -1, 0, or 1 depending on whether x is less than, equal to, or greater than y, respectively.
# complex(r, [i]): Convert real numbers or arrays to complex. i defaults to zero.
# conj(z): Compute the complex conjugate of a complex number z.
# copysign(x, y) -> z: Return z which has the magnitude of x and the same sign as y. (TODO)
# count_ones(x::Integer): Number of ones in the binary representation of x.
# count_zeros(x::Integer): Number of zeros in the binary representation of x.
# denominator(x): Denominator of the rational representation of x.
@zerograd div(x,y)
# divrem(x,y): The quotient and remainder from Euclidean division. Equivalent to (div(x,y), rem(x,y)) or (x÷y, x%y).
@zerograd eps(x)
# factorial,
# fld,
# fld1,
# fldmod,
# fldmod1,
# flipsign,
@primitive float(x),dy dy
# tryparse,
@zerograd floor(x)
# fma,
# gcd,
# gcdx,
# imag,
# inv,
# invmod,
# isapprox,
# iseven,
@zerograd isfinite(x)
@zerograd isinf(x)
@zerograd isinteger(x)
@zerograd isnan(x)
# isodd,
# ispow2,
# isqrt,
# isreal,
# issubnormal,
# iszero,
# isone,
# lcm,
# leading_ones,
# leading_zeros,
# maxintfloat,
# mod,
# mod1,
# muladd,
# nextfloat,
# nextpow,
# nextpow2,
# nextprod,
# numerator,
@zerograd one(x)
# oneunit,
# powermod,
# prevfloat,
# prevpow,
# prevpow2,
# rationalize,
# real,
# realmax,
# realmin,
# reim,
# reinterpret,
@primitive rem(x1,x2),dy,y  unbroadcast(x1,dy)  unbroadcast(x2,-dy.*div.(x1,x2))
@zerograd round(x)
@zerograd sign(x)
@zerograd signbit(x)
# signed: Int function
# trailing_ones:  Int function
# trailing_zeros: Int function
@zerograd trunc(x)
@zerograd unsafe_trunc(x)
@zerograd typemax(x)
@zerograd typemin(x)
# unsigned: Int function
@primitive widemul(x1,x2),dy  unbroadcast(x1,dy.*x2)  unbroadcast(x2,x1.*dy)
@zerograd zero(x)
# √  Same as sqrt
# ∛  Same as cbrt
# ≈  Same as isapprox
# ≉  This is equivalent to !isapprox(x,y) (see isapprox).

### arrays
@zerograd axes(x,i...)
# broadcast!,
# broadcast  Handled in broadcast.jl
# cat  Handled in cat.jl
@zerograd checkbounds(x,i...)
# checkindex,
# circcopy!,
# circshift,
# circshift!,
# clamp!,
# conj!,
# copy!,
# copyto!,
# diff,
# cumprod,
# cumprod!,
# cumsum,
# cumsum!,
# accumulate,
# accumulate!,
@zerograd eachindex(x,i...)
# extrema,
# fill!,
# fill,
# first,
# hcat  Handled in cat.jl
# hvcat,
# indexin,
# argmax,
# argmin,
# invperm,
# invpermute!,
@zerograd isassigned(x,i...)
# isperm,
# issorted,
# last,
# mapslices,
# maximum!,
@primitive maximum(x;d...),dy,y  (dy.*(y.==x))
@primitive maximum(f::typeof(abs),x;d...),dy,y  nothing  (dy.*(y.==abs.(x)).*sign.(x))
# minimum!,
@primitive minimum(x;d...),dy,y  (dy.*(y.==x))
@primitive minimum(f::typeof(abs),x;d...),dy,y  nothing  (dy.*(y.==abs.(x)).*sign.(x))
# minmax,
@zerograd ndims(x)
@zerograd ones(x)
# parent,
# parentindices,
# partialsort,
# partialsort!,
# partialsortperm,
# partialsortperm!,
# permute!,
@primitive permutedims(x,d...),dy  permutedims(dy,invperm(d...))
# permutedims!,
# prod!,
@primitive prod(x;d...),dy,y  (dy.*(y./x))  # TODO: prod with abs, abs2
# promote_shape,
# range,
@primitive reshape(x,i...),dy  reshape(dy,size(x))
# reverse!,
# reverse,
# rot180,
# rotl90,
# rotr90,
# shuffle,
# shuffle!,
@zerograd size(x,i...)
# selectdim, # implemented in getindex.jl
# sort!,
# sort,
# sortcols,
# sortperm,
# sortperm!,
# sortrows,
# squeeze,
# step,
@zerograd stride(x,i...)
@zerograd strides(x)
# sum!,
@primitive sum(x;d...),dy  (dy.*one.(x))
@primitive sum(f::typeof(abs),x;d...),dy   nothing  (dy.*sign.(x))
@primitive sum(f::typeof(abs2),x;d...),dy  nothing  (dy.*(2x))
# to_indices,
# vcat  Handled in cat.jl
@primitive vec(x),dy  reshape(dy,size(x))
# view, # implemented in getindex.jl
# zeros: no longer called with array arguments, using zero instead.


# collections
# all!,
@zerograd all(a;dims=:)
@zerograd all(f,a;dims=:)
# allunique,
# any!,
@zerograd any(a;dims=:)
@zerograd any(f,a;dims=:)
# firstindex,
# collect,
@zerograd count(a;dims=:)
@zerograd count(f,a;dims=:)
# delete!,
# deleteat!,
@zerograd eltype(x)
# empty!,
# empty,
@zerograd lastindex(x,d...)
# filter!,
# filter,
# foldl,
# foldr,
# foreach,
# get  Handled in getindex.jl
# get!,
# getindex  Handled in getindex.jl
# getkey,
# haskey,
# in,
# intersect!,
# intersect,
@zerograd isempty(x)
# issubset,
# issetequal,
# keys,
# keytype,
@zerograd length(x)
# map!,
# map,
# mapfoldl,
# mapfoldr,
# mapreduce,
# merge!,
# merge,
# pairs,
# reduce,
# setdiff!,
# setdiff,
# setindex!,
@zerograd similar(x,i...)
similar(x::Value, dims::Base.DimOrInd...) = similar(value(x), dims...) # 0.7 ambiguity fix
# sizehint!,
# splice!,
# symdiff!,
# symdiff,
# union!,
# union,
# unique!,
# unique,
values(a::Value{T})  where {T<:AbstractDict} = (a[k] for k in keys(value(a)))
# valtype,
# ∈,
# ∉,
# ∋,
# ∌,
# ⊆,
# ⊈,
# ⊊,
# ⊇,
# ⊉,
# ⊋,
# ∩,
# ∪,

### iteration
# done  deprecated
# next  deprecated
# start deprecated
# iterate    Handled in iterate.jl
# enumerate  Implemented with iterate
# zip        Implemented with iterate

### object identity and equality
@primitive copy(x),dy dy
# deepcopy,
# hash,
@primitive identity(x),dy dy
# isbits,
@zerograd isequal(x,y)
# isimmutable,
@zerograd isless(x,y)
# ifelse,
# objectid,
# sizeof,

### types
# convert,
# getproperty,
# setproperty!,
# fieldoffset,
# fieldname,
# fieldnames,
# fieldcount,
# propertynames,
# isabstracttype,
# isbitstype,
# isprimitivetype,
# isstructtype,
# isconcretetype,
# isdispatchtuple,
@primitive oftype(t,x),dy nothing oftype(x,dy)
# promote,
# promote_rule,
# promote_type,
# instances,
# supertype,
# typeintersect,
# typejoin,
# widen,
