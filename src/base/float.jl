float1zero = [
:ceil,                     # float,operators
:floor,                    # float,operators
:isfinite,                 # float,operators
:isinf,                    # float,operators
:isnan,                    # float,operators
:round,                    # float,operators
:trunc,                    # float,operators
]
for f in float1zero; @eval @zerograd $f(x::AorN); end

float2zero = [
:div,                       # The quotient from Euclidean division. Computes x/y, truncated to an integer; operators
#:÷, 			 # Same as div.
]
for f in float2zero; @eval @zerograd $f(x1::AorN,x2::AorN); end

float1arg = Dict{Symbol,Any}(
:(+) => :identity,  # supports (N,) (A,) (N,N) (N,A) (A,N) (A,A); float arraymath abstractarraymath operators
:(-) => :(-),  # supports (N,) (A,) (N,N) (N,A) (A,N) (A,A); float arraymath
:abs => :(dy->dy.*sign(x)),             # float,operators
:float => :identity,                    # float
)

for (f,g) in float1arg
    @eval @primitive $f(x::AorN)::y $g
end

float2arg = Dict{Symbol,Any}(
:+ => (:identity,:identity), # extra (N,) (A,); float,arraymath,abstractarraymath,operators
:- => (:identity,:(-)),                    # extra (N,) (A,); float,arraymath
#:rem => (1,:(-trunc(x1./x2))),   # Remainder from Euclidean division, return same sign as x, BUG: missing (A,A); BUG: WARNING: (:check_grads,(:sum,:rem),:args,(-0.13412338383912367,[-0.00025363687477246275,0.4389355563026644]),:exact,(2.0,[-528.0,0.0]),:numeric,(1.9999999999997797,[0.8920182562441314,0.0])); float,arraymath
#:% => (1,:(-trunc(x1./x2))),     # same as rem
#:mod => (1,:(-floor(x1./x2))),   # BUG: WARNING: (:check_grads,(:sum,:mod),:args,([-1.850960311615227,-1.282613199024709],[0.8035558268314972,-0.32067619631949534]),:exact,([1.0,1.0],[3.0,-3.0]),:numeric,([1.0000000000021103,1.0000000000021103],[2.9999999999996696,3203.2619631949538])); Modulus after division, return same sign as y; float,arraymath
#:(==) => 0,                      # BUG: StackOverflowError(). supports any pair of values; float,operators,abstractarray
)

for (f,g) in float2arg
    @eval @primitive $f(x1::AorN,x2::AorN)::y unbroadcast(y,x1,$(g[1])) unbroadcast(y,x2,$(g[2]))
end

# defgrads(float2arg, Number, Number)
# defgrads(float2arg, AbstractArray, Number)
# defgrads(float2arg, Number, AbstractArray)
# defgrads(float2arg, AbstractArray, AbstractArray)

# Methods for multiplication:
# *(x::Float64, y::Float64) at float.jl:212  (same as .*)
# *(A::Number, B::AbstractArray{T,N}) at abstractarraymath.jl:54  (calls .*)
# *(A::AbstractArray{T,N}, B::Number) at abstractarraymath.jl:55  (calls .*)
# The Array-Array case is handled by linalg/matmul.
# float2mul = Dict{Symbol,Any}(
# :* => (:x2,:x1),                   # (N,) (M,) (N,*) (*,N) (M,V) (M,M)
# )                             

# defgrads(float2mul, Number, Number)
# defgrads(float2mul, AbstractArray, Number)
# defgrads(float2mul, Number, AbstractArray)

@primitive (*)(x1::Number,x2::Number) (dy->dy.*x2) (dy->dy.*x1)
@primitive (*)(x1::Number,x2::AbstractArray)::y unbroadcast(y,x1,(dy->dy.*x2)) (dy->dy.*x1)
@primitive (*)(x1::AbstractArray,x2::Number)::y (dy->dy.*x2) unbroadcast(y,x2,(dy->dy.*x1))

# Methods for division:
# /(x::Float64, y::Float64) at float.jl:214
# /(A::AbstractArray{T,N}, B::Number) at abstractarraymath.jl:57
# The Array-Array case is handled by linalg/generic.

# float2div = Dict{Symbol,Any}(
# :/ => (:(1./x2),:(-x1./abs2(x2))), # (N,N) (A,N)
# )                             

# defgrads(float2div, Number, Number)
# defgrads(float2div, AbstractArray, Number)

@primitive (/)(x1::Number,x2::Number) (dy->dy./x2) (dy->-dy.*x1./abs2(x2))
@primitive (/)(x1::AbstractArray,x2::Number)::y (dy->dy./x2) unbroadcast(y,x2,(dy->-dy.*x1./abs2(x2)))

# These are defined in terms of isless which is handled in interfaces.jl
# float2arg1 = Dict{Symbol,Any}(
# :< => 0,                         # only supports (N,N), arrays not supported; float,operators
# :<= => 0,                        # only supports (N,N), arrays not supported; float,operators
# :> => 0,                         # only supports (N,N), arrays not supported; operators
# :>= => 0,                        # only supports (N,N), arrays not supported; operators
# )

#BUG defgrads(float2arg1, Number, Number) # causes segfault?

# These operators take bool or int as input and are excluded:

# bool1arg = Dict{Symbol,Any}(
# #:(~) => :todo, # bitwise not, domain=Integer; bool,int,gmp,bitarray,arraymath
# #:(!) => :todo, # boolean not, domain=Bool; bool,bitarray,arraymath
# )

# int1arg = Dict{Symbol,Any}(
# #:(~) => :todo, # bitwise not, domain=Integer; bool,int,gmp,bitarray,arraymath
# )
# int2arg = Dict{Symbol,Any}(
# #:$ => :todo,                     # domain: Integers, bitwise xor; int,bool,bitarray,gmp,operators,promotion,arraymath,sparsematrix
# #:& => :todo,                     # domain: Integers, bitwise and
# #:| => :todo,                     # domain: Integers, bitwise or
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
