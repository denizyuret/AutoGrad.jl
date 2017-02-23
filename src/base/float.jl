float1zero = [
:ceil,
:floor,
:isfinite,
:isinf,
:isnan,
:round,
:trunc,
]
for f in float1zero; @eval @zerograd $f(x); end

float2zero = [
:div, # The quotient from Euclidean division. Computes x/y, truncated to an integer; operators
#:÷,  # Same as div.
]
for f in float2zero; @eval @zerograd $f(x1,x2); end

float1arg = [
(:+, :dy),  # supports (N,) (A,) (N,N) (N,A) (A,N) (A,A); float arraymath abstractarraymath operators
(:-, :(-dy)),  # supports (N,) (A,) (N,N) (N,A) (A,N) (A,A); float arraymath
(:float, :dy),          # float
]

for (f,g) in float1arg
    @eval @primitive $f(x),dy,y $g
    addtest1(f,(-Inf,Inf))
end

float2arg = [
(:+, :dy, :dy), # extra (N,) (A,); float,arraymath,abstractarraymath,operators
(:-, :dy, :(-dy)),                    # extra (N,) (A,); float,arraymath
#(:rem, :dy, :(-dy.*trunc(x1./x2))),   # Remainder from Euclidean division, return same sign as x, BUG: missing (A,A); BUG: WARNING: (:check_grads,(:sum,:rem),:args,(-0.13412338383912367,[-0.00025363687477246275,0.4389355563026644]),:exact,(2.0,[-528.0,0.0]),:numeric,(1.9999999999997797,[0.8920182562441314,0.0])); float,arraymath
#(:%, :dy, :(-dy.*trunc(x1./x2))),     # same as rem
#(:mod, :dy, :(-dy.*floor(x1./x2))),   # BUG: WARNING: (:check_grads,(:sum,:mod),:args,([-1.850960311615227,-1.282613199024709],[0.8035558268314972,-0.32067619631949534]),:exact,([1.0,1.0],[3.0,-3.0]),:numeric,([1.0000000000021103,1.0000000000021103],[2.9999999999996696,3203.2619631949538])); Modulus after division, return same sign as y; float,arraymath
#(:(==), 0,                      # BUG: StackOverflowError(). supports any pair of values; float,operators,abstractarray
]

for (f,g1, g2) in float2arg
    @eval @primitive $f(x1,x2),dy,y unbroadcast(x1,$g1) unbroadcast(x2,$g2)
    addtest2(f,(-Inf,Inf))
end

# Methods for multiplication:
# *(x::Float64, y::Float64) at float.jl:212  (same as .*)
# *(A::Number, B::AbstractArray{T,N}) at abstractarraymath.jl:54  (calls .*)
# *(A::AbstractArray{T,N}, B::Number) at abstractarraymath.jl:55  (calls .*)
# The Array-Array case is handled by linalg/matmul.
# a*b' etc. get handled by A_mul_Bc even if a or b are scalar.
# So we'll handle all multiplication in matmul.jl.

# Methods for division:
# /(x::Float64, y::Float64) at float.jl:214
# /(A::AbstractArray{T,N}, B::Number) at abstractarraymath.jl:57
# The Array-Array case is handled by linalg/generic.
# There is no Number-Array support.
# \(x,A) is the same as /(A,x)

@primitive (/)(x1,x2::Number),dy,y  (dy/x2)  unbroadcast(x2,-dy.*x1./abs2(x2))
x = randn(); a = randn(2)
addtest(/,randn(),randn())
addtest(/,randn(2),randn())

@primitive (\)(x2::Number,x1),dy,y  unbroadcast(x2,-dy.*x1./abs2(x2))  (dy/x2)
addtest(\,randn(),randn())
addtest(\,randn(),randn(2))

# These are defined in terms of isless which is handled in interfaces.jl
# float2arg1 = Dict{Symbol,Any}(
# :<, 0,                         # only supports (N,N), arrays not supported; float,operators
# :<=, 0,                        # only supports (N,N), arrays not supported; float,operators
# :>, 0,                         # only supports (N,N), arrays not supported; operators
# :>=, 0,                        # only supports (N,N), arrays not supported; operators
# )

# These operators take bool or int as input and are excluded:

# bool1arg = Dict{Symbol,Any}(
# #:(~), :todo, # bitwise not, domain=Integer; bool,int,gmp,bitarray,arraymath
# #:(!), :todo, # boolean not, domain=Bool; bool,bitarray,arraymath
# )

# int1arg = Dict{Symbol,Any}(
# #:(~), :todo, # bitwise not, domain=Integer; bool,int,gmp,bitarray,arraymath
# )
# int2arg = Dict{Symbol,Any}(
# #:$, :todo,                     # domain: Integers, bitwise xor; int,bool,bitarray,gmp,operators,promotion,arraymath,sparsematrix
# #:&, :todo,                     # domain: Integers, bitwise and
# #:|, :todo,                     # domain: Integers, bitwise or
# )


# TODO:

# eval
# convert
# promote_rule
# float
# unsafe_trunc
# trunc
# floor
# ceil
# round
# widen
# -
# +
# *
# /
# muladd
# rem
# cld
# mod
# ==
# !=
# <
# <=
# isequal
# isless
# cmp
# abs
# isnan
# isfinite
# isinf
# hx: Not exported
# hash
# precision
# float_lex_order: Not exported
# nextfloat
# prevfloat
# issubnormal
# typemin
# typemax
# realmin
# realmax
# eps
# bswap
# reinterpret
# sign_mask: Not exported
# exponent_mask: Not exported
# exponent_one: Not exported
# exponent_half: Not exported
# significand_mask: Not exported
# significand_bits: Not exported
# exponent_bits: Not exported
# exponent_bias: Not exported
# $(Expr(:$, :fn)): Not a symbol
# big
# (abs, :(dy.*sign(x))) moved to number.jl

