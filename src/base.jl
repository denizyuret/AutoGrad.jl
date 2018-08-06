# coverage
# ./util.jl		--done
# ./interfaces.jl	--done
# ./base/abstractarray.jl --> cat.jl, done
# ./base/abstractarraymath.jl --done
# ./base/arraymath.jl	--done
# ./base/broadcast.jl	--done
# ./base/float.jl	--done
# ./base/math.jl	--done
# ./base/number.jl	--done
# ./base/reduce.jl	--done
# ./base/statistics.jl	--done
# ./linalg/dense.jl
# ./linalg/generic.jl
# ./linalg/matmul.jl
# ./special/bessel.jl	--done
# ./special/erf.jl      --done
# ./special/gamma.jl    --done
# ./special/trig.jl	--done


# Used deps/imports.pl to generate the next line
import Base: !=, !==, *, +, -, /, <, <=, ==, >, >=, \, ^, abs, abs2, acos, acosd, acosh, acot, acotd, acoth, acsc, acscd, acsch, adjoint, all, any, asec, asecd, asech, asin, asind, asinh, atan, atand, atanh, big, cbrt, ceil, checkbounds, clamp, copy, cos, cosc, cosd, cosh, cospi, cot, cotd, coth, count, csc, cscd, csch, deg2rad, div, eachindex, eltype, eps, exp, exp10, exp2, expm1, exponent, float, floor, hypot, isassigned, isempty, isequal, isfinite, isinf, isinteger, isless, isnan, lastindex, ldexp, length, log, log10, log1p, log2, max, maximum, min, minimum, mod2pi, ndims, one, ones, permutedims, prod, rad2deg, rem, reshape, round, sec, secd, sech, sign, signbit, significand, similar, sin, sinc, sind, sinh, sinpi, size, sqrt, stride, strides, sum, tan, tand, tanh, transpose, trunc, vec, zero, zeros

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
@primitive *(x),dy  dy
@primitive1 broadcast(f::typeof(*),x1,x2),dy  nothing  unbroadcast(x1,dy.*x2)  unbroadcast(x2,x1.*dy)
@primitive1 *(x1::Number,x2::Number),dy                unbroadcast(x1,dy.*x2)  unbroadcast(x2,x1.*dy)
@primitive1 *(x1::Number,x2),dy                        unbroadcast(x1,dy.*x2)  unbroadcast(x2,x1.*dy)
@primitive1 *(x1,x2::Number),dy                        unbroadcast(x1,dy.*x2)  unbroadcast(x2,x1.*dy)
@primitive1 *(x1,x2),dy  (dy*x2')  (x1'*dy)
@primitive +(x),dy  dy
@primitive +(x1,x2),dy  unbroadcast(x1,dy)  unbroadcast(x2,dy)
@primitive -(x),dy  -dy
@primitive -(x1,x2),dy  unbroadcast(x1,dy)  unbroadcast(x2,-dy)
@primitive1 broadcast(f::typeof(/),x1,x2),dy  nothing  unbroadcast(x1,dy./x2)  unbroadcast(x2,-dy.*x1./abs2.(x2))
@primitive1 /(x1::Number,x2::Number),dy                unbroadcast(x1,dy./x2)  unbroadcast(x2,-dy.*x1./abs2.(x2))
@primitive1 /(x1::Number,x2),dy                        unbroadcast(x1,dy./x2)  unbroadcast(x2,-dy.*x1./abs2.(x2))
@primitive1 /(x1,x2::Number),dy                        unbroadcast(x1,dy./x2)  unbroadcast(x2,-dy.*x1./abs2.(x2))
# @primitive1 /(x1,x2),dy # TODO for array arguments without broadcast
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
@primitive1 broadcast(f::typeof(\),x1,x2),dy  nothing  unbroadcast(x1,-dy.*x2./abs2.(x1))  unbroadcast(x2,dy./x1)
@primitive1 \(x1::Number,x2::Number),dy                unbroadcast(x1,-dy.*x2./abs2.(x1))  unbroadcast(x2,dy./x1)
@primitive1 \(x1::Number,x2),dy                        unbroadcast(x1,-dy.*x2./abs2.(x1))  unbroadcast(x2,dy./x1)
@primitive1 \(x1,x2::Number),dy                        unbroadcast(x1,-dy.*x2./abs2.(x1))  unbroadcast(x2,dy./x1)
# @primitive1 \(x1,x2),dy # TODO for array arguments without broadcast
@primitive ^(x1,x2),dy,y  unbroadcast(x1,dxndx(x1,x2,dy))  unbroadcast(x2,dy.*y.*log.(x1))
@primitive ^(x1,x2::Integer),dy,y  unbroadcast(x1,dxndx(x1,x2,dy))  unbroadcast(x2,dy.*y.*log.(x1)) # ambiguity fix
dxndx(x1,x2,dy)=(if x2==0; zero(dy); elseif x2==1; dy; elseif x2==2; 2x1.*dy; else; dy.*x2.*x1.^(x2-1); end) # optimize common cases
# |   Int function
# |>  Function chaining
# ~   Int function
# :   Range operator
# =>  Pair constructor
# ∘   Function composition

### scalar math
# @evalpoly(z,c...): Evaluate the polynomial \sum_k c[k] z^{k-1}.
@primitive abs(x),dy (dy.*sign.(x))
@primitive abs2(x),dy (dy.*(2x))
@primitive acos(x),dy (dy.*(-1 ./ sqrt.(1 - abs2.(x))))
@primitive acosd(x),dy (dy.*(-(180 / pi) ./ sqrt.(1 - abs2.(x))))
@primitive acosh(x),dy (dy.*(1 ./ sqrt.(abs2.(x) - 1)))
@primitive acot(x),dy (dy.*(-1 ./ (1 + abs2.(x))))
@primitive acotd(x),dy (dy.*(-(180 / pi) ./ (1 + abs2.(x))))
@primitive acoth(x),dy (dy.*(1 ./ (1 - abs2.(x))))
@primitive acsc(x),dy (dy.*(-1 ./ sqrt.(((x .* x) .* (x - 1)) .* (x + 1))))
@primitive acscd(x),dy (dy.*(-(180 / pi) ./ sqrt.(((x .* x) .* (x - 1)) .* (x + 1))))
@primitive acsch(x),dy (dy.*(-1 ./ sqrt.(x .^ 4 + x .^ 2)))
# angle(z): Compute the phase angle in radians of a complex number z.
@primitive asec(x),dy (dy.*(1 ./ sqrt.(x .^ 4 - x .^ 2)))
@primitive asecd(x),dy (dy.*((180 / pi) ./ sqrt.(x .^ 4 - x .^ 2)))
@primitive asech(x),dy (dy.*(-1 ./ sqrt.(x .^ 2 - x .^ 4)))
@primitive asin(x),dy (dy.*(1 ./ sqrt.(1 - abs2.(x))))
@primitive asind(x),dy (dy.*((180 / pi) ./ sqrt.(1 - abs2.(x))))
@primitive asinh(x),dy (dy.*(1 ./ sqrt.(1 + abs2.(x))))
@primitive atan(x),dy (dy.*(1 ./ (1 + abs2.(x))))
@primitive atan(x1,x2),dy,y  unbroadcast(x1,dy.*(x2./(abs2.(x1)+abs2.(x2))))  unbroadcast(x2,dy.*(-x1./(abs2.(x1)+abs2.(x2))))
@primitive atand(x),dy (dy.*((180 / pi) ./ (1 + abs2.(x))))
@primitive atanh(x),dy (dy.*(1 ./ (1 - abs2.(x))))
@primitive big(x),dy oftype(x,dy)
# binomial(n,k): Number of ways to choose k out of n items.
# bswap(n): Reverse the byte order of n.
@primitive cbrt(x),dy,y (dy.*(1 ./ (3 .* abs2.(y))))
@zerograd ceil(x)
# cis(z): Return \exp(iz).
@primitive clamp(x,lo,hi),dy,y  unbroadcast(x,dy.*(lo .<= x .<= hi))
# cld(x, y): Smallest integer larger than or equal to x/y.
# cmp(x,y): Return -1, 0, or 1 depending on whether x is less than, equal to, or greater than y, respectively.
# complex(r, [i]): Convert real numbers or arrays to complex. i defaults to zero.
# conj(z): Compute the complex conjugate of a complex number z.
# copysign(x, y) -> z: Return z which has the magnitude of x and the same sign as y. (TODO)
@primitive cos(x),dy (dy.*(-(sin.(x))))
@primitive cosc(x),dy (dy.*((-2y) ./ x - sinc.(x) * pi ^ 2))
@primitive cosd(x),dy (dy.*((-(sind.(x)) * pi) / 180))
@primitive cosh(x),dy (dy.*(sinh.(x)))
@primitive cospi(x),dy (dy.*(-(sinpi.(x)) * pi))
@primitive cot(x),dy (dy.*(-(abs2.(csc.(x)))))
@primitive cotd(x),dy (dy.*((-(abs2.(cscd.(x))) * pi) / 180))
@primitive coth(x),dy (dy.*(-(abs2.(csch.(x)))))
# count_ones(x::Integer): Number of ones in the binary representation of x.
# count_zeros(x::Integer): Number of zeros in the binary representation of x.
@primitive csc(x),dy,y (dy.*(-y .* cot.(x)))
@primitive cscd(x),dy,y (dy.*(((-y .* cotd.(x)) * pi) / 180))
@primitive csch(x),dy,y (dy.*(-y .* coth.(x)))
@primitive deg2rad(x),dy (dy.*(pi / 180))
# denominator(x): Denominator of the rational representation of x.
@zerograd div(x,y)
# divrem(x,y): The quotient and remainder from Euclidean division. Equivalent to (div(x,y), rem(x,y)) or (x÷y, x%y).
@zerograd eps(x)
@primitive exp(x),dy,y (dy.*(y))
@primitive exp10(x),dy,y (dy.*(y .* log(10)))
@primitive exp2(x),dy,y (dy.*(y .* log(2)))
@primitive expm1(x),dy,y (dy.*(1 + y))
@zerograd exponent(x)
#     factorial,
#     fld,
#     fld1,
#     fldmod,
#     fldmod1,
#     flipsign,
@primitive float(x),dy dy
#     tryparse,
@zerograd floor(x)
#     fma,
#     frexp,
#     gcd,
#     gcdx,
#     hypot,
@primitive hypot(x1,x2),dy,y  unbroadcast(x1,dy.*(x1./y))  unbroadcast(x2,dy.*(x2./y))
#     imag,
#     inv,
#     invmod,
#     isapprox,
#     iseven,
@zerograd isfinite(x)
@zerograd isinf(x)
@zerograd isinteger(x)
@zerograd isnan(x)
#     isodd,
#     ispow2,
#     isqrt,
#     isreal,
#     issubnormal,
#     iszero,
#     isone,
#     lcm,
@primitive ldexp(x,n),dy  (dy.*(2 .^ n))
#     leading_ones,
#     leading_zeros,
@primitive log(x),dy (dy.*(1 ./ x))
@primitive log(x1,x2),dy,y  unbroadcast(x1,dy.*(-log.(x2)./(x1.*abs2.(log.(x1)))))  unbroadcast(x2,dy.*(1 ./ (x2.*log.(x1))))
@primitive log10(x),dy (dy.*(1 ./ (log(10) .* x)))
@primitive log1p(x),dy (dy.*(1 ./ (1 + x)))
@primitive log2(x),dy (dy.*(1 ./ (log(2) .* x)))
#     maxintfloat,
#     mod,
#     mod1,
#     modf,
@primitive mod2pi(x),dy dy
#     muladd,
#     nextfloat,
#     nextpow,
#     nextpow2,
#     nextprod,
#     numerator,
@zerograd one(x)
#     oneunit,
#     powermod,
#     prevfloat,
#     prevpow,
#     prevpow2,
@primitive rad2deg(x),dy (dy.*(180 / pi))
#     rationalize,
#     real,
#     realmax,
#     realmin,
#     reim,
#     reinterpret,
@primitive rem(x1,x2),dy,y  unbroadcast(x1,dy)  unbroadcast(x2,-dy.*div.(x1,x2))
#     rem2pi,
@zerograd round(x)
@primitive sec(x),dy,y (dy.*(y .* tan.(x)))
@primitive secd(x),dy,y (dy.*(((y .* tand.(x)) * pi) / 180))
@primitive sech(x),dy,y (dy.*(-y .* tanh.(x)))
@zerograd sign(x)
@zerograd signbit(x)
#     signed,
@primitive significand(x),dy (dy.*(0.5 .^ exponent.(x)))
@primitive sin(x),dy (dy.*(cos.(x)))
@primitive sinc(x),dy (dy.*(cosc.(x)))
#     sincos,
@primitive sind(x),dy (dy.*((cosd.(x) * pi) / 180))
@primitive sinh(x),dy (dy.*(cosh.(x)))
@primitive sinpi(x),dy (dy.*(cospi.(x) * pi))
@primitive sqrt(x),dy,y (dy.*(1 ./ (2 .* y)))
@primitive tan(x),dy,y (dy.*(1 + abs2.(y)))
@primitive tand(x),dy,y (dy.*(((1 + abs2.(y)) * pi) / 180))
@primitive tanh(x),dy,y (dy.*(1 - abs2.(y)))
#     trailing_ones,
#     trailing_zeros,
@zerograd trunc(x)
#     unsafe_trunc,
#     typemax,
#     typemin,
#     unsigned,
#     widemul,
@zerograd zero(x)
#     √,
#     ∛,
#     ≈,
#     ≉,

### arrays
#     axes,
#     broadcast!,
#     broadcast,
# cat  Handled in cat.jl
@zerograd checkbounds(x,i...)
#     checkindex,
#     circcopy!,
#     circshift,
#     circshift!,
#     clamp!,
#     conj!,
#     copy!,
#     copyto!,
#     diff,
#     cumprod,
#     cumprod!,
#     cumsum,
#     cumsum!,
#     accumulate,
#     accumulate!,
@zerograd eachindex(x,i...)
#     extrema,
#     fill!,
#     fill,
#     first,
# hcat  Handled in cat.jl
#     hvcat,
#     indexin,
#     argmax,
#     argmin,
#     invperm,
#     invpermute!,
@zerograd isassigned(x,i...)
#     isperm,
#     issorted,
#     last,
#     mapslices,
@primitive max(x1,x2),dy,y  unbroadcast(x1,dy.*(y.==x1))  unbroadcast(x2,dy.*(y.==x2))
#     maximum!,
@primitive maximum(x;d...),dy,y  (dy.*(y.==x))
@primitive maximum(f::typeof(abs),x;d...),dy,y  nothing  (dy.*(y.==abs.(x)).*sign.(x))
@primitive min(x1,x2),dy,y  unbroadcast(x1,dy.*(y.==x1))  unbroadcast(x2,dy.*(y.==x2))
#     minimum!,
@primitive minimum(x;d...),dy,y  (dy.*(y.==x))
@primitive minimum(f::typeof(abs),x;d...),dy,y  nothing  (dy.*(y.==abs.(x)).*sign.(x))
#     minmax,
@zerograd ndims(x)
@zerograd ones(x)
#     parent,
#     parentindices,
#     partialsort,
#     partialsort!,
#     partialsortperm,
#     partialsortperm!,
#     permute!,
@primitive permutedims(x,d...),dy  permutedims(dy,invperm(d...))
#     permutedims!,
#     prod!,
@primitive prod(x),dy,y  (dy.*(y./x))  # TODO: prod with abs, abs2
#     promote_shape,
#     range,
@primitive reshape(x,i...),dy  reshape(dy,size(x))
#     reverse!,
#     reverse,
#     rot180,
#     rotl90,
#     rotr90,
#     shuffle,
#     shuffle!,
@zerograd size(x,i...)
#     selectdim,
#     sort!,
#     sort,
#     sortcols,
#     sortperm,
#     sortperm!,
#     sortrows,
#     squeeze,
#     step,
@zerograd stride(x,i...)
@zerograd strides(x)
#     sum!,
@primitive sum(x;d...),dy  (dy.*one.(x))
@primitive sum(f::typeof(abs),x;d...),dy   nothing  (dy.*sign.(x))
@primitive sum(f::typeof(abs2),x;d...),dy  nothing  (dy.*(2x))
#     to_indices,
# vcat  Handled in cat.jl
@primitive vec(x),dy  reshape(dy,size(x))
#     view,
@zerograd zeros(x)

# linear algebra: Move these to LinearAlgebra?
@primitive adjoint(x),dy    adjoint(dy)
@primitive transpose(x),dy  transpose(dy)
#     kron,

# collections
#     all!,
@zerograd all(a;dims=:)
@zerograd all(f,a;dims=:)
#     allunique,
#     any!,
@zerograd any(a;dims=:)
@zerograd any(f,a;dims=:)
#     firstindex,
#     collect,
@zerograd count(a;dims=:)
@zerograd count(f,a;dims=:)
#     delete!,
#     deleteat!,
@zerograd eltype(x)
#     empty!,
#     empty,
@zerograd lastindex(x)
#     filter!,
#     filter,
#     foldl,
#     foldr,
#     foreach,
# get  Handled in getindex.jl
#     get!,
# getindex  Handled in getindex.jl
#     getkey,
#     haskey,
#     in,
#     intersect!,
#     intersect,
@zerograd isempty(x)
#     issubset,
#     issetequal,
#     keys,
#     keytype,
@zerograd length(x)
#     map!,
#     map,
#     mapfoldl,
#     mapfoldr,
#     mapreduce,
#     merge!,
#     merge,
#     pairs,
#     reduce,
#     setdiff!,
#     setdiff,
#     setindex!,
@zerograd similar(x,i...)
#     sizehint!,
#     splice!,
#     symdiff!,
#     symdiff,
#     union!,
#     union,
#     unique!,
#     unique,
#     values,
#     valtype,
#     ∈,
#     ∉,
#     ∋,
#     ∌,
#     ⊆,
#     ⊈,
#     ⊊,
#     ⊇,
#     ⊉,
#     ⊋,
#     ∩,
#     ∪,

### iteration
# done  deprecated
# next  deprecated
# start deprecated
# iterate    Handled in iterate.jl
# enumerate  Implemented with iterate
# zip        Implemented with iterate

### object identity and equality
@primitive copy(x),dy dy
#     deepcopy,
#     hash,
#     identity,
#     isbits,
@zerograd isequal(x,y)
#     isimmutable,
@zerograd isless(x,y)
#     ifelse,
#     objectid,
#     sizeof,
