# julia/base/exports.jl
import Base: \, +, -, /, <, <=, ==, >, >=, abs, abs2, acos, acosd, acosh, acot, acotd, acoth, acsc, acscd, acsch, asec, asecd, asech, asin, asind, asinh, atan, atand, atanh, big, cbrt, ceil, checkbounds, clamp, copy, cos, cosc, cosd, cosh, cospi, cot, cotd, coth, csc, cscd, csch, deg2rad, div, eachindex, eltype, eps, exp, exp10, exp2, expm1, exponent, float, floor, isassigned, isempty, isequal, isfinite, isinf, isless, isnan, lastindex, length, log, log10, log1p, log2, ndims, one, ones, rad2deg, reshape, round, sec, secd, sech, significand, similar, sin, sinc, sind, sinh, sinpi, size, sqrt, stride, strides, tan, tand, tanh, trunc, zero, zeros

# coverage
# ./interfaces.jl
# ./base/abstractarray.jl
# ./base/abstractarraymath.jl
# ./base/arraymath.jl
# ./base/broadcast.jl
# ./base/float.jl
# ./base/math.jl
# ./base/number.jl
# ./base/reduce.jl
# ./base/statistics.jl
# ./linalg/dense.jl
# ./linalg/generic.jl
# ./linalg/matmul.jl
# ./special/bessel.jl
# ./special/erf.jl
# ./special/gamma.jl
# ./special/trig.jl


# Operators
#     !,
#     !=,
#     ≠,
#     !==,
#     ≡,
#     ≢,
#     xor,
#     ⊻,
#     %,
#     ÷,
#     &,
#     *,
@primitive +(x),dy  dy
@primitive +(x1,x2),dy  unbroadcast(x1,dy)  unbroadcast(x2,dy)
@primitive -(x),dy  -dy
@primitive -(x1,x2),dy  unbroadcast(x1,dy)  unbroadcast(x2,-dy)
@primitive /(x1,x2::Number),dy,y  (dy/x2)  unbroadcast(x2,-dy.*x1./abs2.(x2))
#     //,
@zerograd <(x1,x2)
#     <:,
#     <<,
@zerograd <=(x1,x2)
#     ≤,
@zerograd ==(x1,x2)
@zerograd >(x1,x2)
#     >:,
@zerograd >=(x1,x2)
#     ≥,
#     >>,
#     >>>,
@primitive \(x2::Number,x1),dy,y  unbroadcast(x2,-dy.*x1./abs2.(x2))  (dy/x2)
#     ^,
#     |,
#     |>,
#     ~,
#     :,
#     =>,
#     ∘,

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
#     imag,
#     inv,
#     invmod,
#     isapprox,
#     iseven,
@zerograd isfinite(x)
@zerograd isinf(x)
#     isinteger,
@zerograd isnan(x)
#     isodd,
#     ispow2,
#     isqrt,
#     isreal,
#     issubnormal,
#     iszero,
#     isone,
#     lcm,
#     ldexp,
#     leading_ones,
#     leading_zeros,
@primitive log(x),dy (dy.*(1 ./ x))
@primitive log10(x),dy (dy.*(1 ./ (log(10) .* x)))
@primitive log1p(x),dy (dy.*(1 ./ (1 + x)))
@primitive log2(x),dy (dy.*(1 ./ (log(2) .* x)))
#     maxintfloat,
#     mod,
#     mod1,
#     modf,
#     mod2pi,
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
#     rem,
#     rem2pi,
@zerograd round(x)
@primitive sec(x),dy,y (dy.*(y .* tan.(x)))
@primitive secd(x),dy,y (dy.*(((y .* tand.(x)) * pi) / 180))
@primitive sech(x),dy,y (dy.*(-y .* tanh.(x)))
#     sign,
#     signbit,
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
#     cat,
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
#     hcat,
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
#     max,
#     maximum!,
#     maximum,
#     min,
#     minimum!,
#     minimum,
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
#     permutedims,
#     permutedims!,
#     prod!,
#     prod,
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
#     sum,
#     to_indices,
#     vcat,
#     vec,
#     view,
@zerograd zeros(x)

# search, find, match and related functions
#     eachmatch,
#     endswith,
#     findall,
#     findfirst,
#     findlast,
#     findmax,
#     findmin,
#     findmin!,
#     findmax!,
#     findnext,
#     findprev,
#     match,
#     occursin,
#     searchsorted,
#     searchsortedfirst,
#     searchsortedlast,
#     startswith,

# linear algebra
#     adjoint,
#     transpose,
#     kron,

# bitarrays
#     falses,
#     trues,

# dequeues
#     append!,
#     insert!,
#     pop!,
#     prepend!,
#     push!,
#     resize!,
#     popfirst!,
#     pushfirst!,

# collections
#     all!,
#     all,
#     allunique,
#     any!,
#     any,
#     firstindex,
#     collect,
#     count,
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
#     get,
#     get!,
#     getindex,
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

# iteration
#     done,
#     next,
#     start,
#     iterate,

#     enumerate,  # re-exported from Iterators
#     zip,

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

# implemented in Random module
#     rand,
#     randn,

