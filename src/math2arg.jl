# Some of these functions come from grepping vectorize_2arg in Base
# (defined in operators, used in fastmath, floatfuncs, math, bessel,
# gamma), which allows them to have Array arguments in first, second,
# or both positions.  When both arguments are Arrays they must have
# the same size, or if one has extra dimensions at the end they need
# to be 1.  The resulting array will have the longer of the two sizes.
# (implemented by promote_shape).  Note that no broadcasting is
# performed here, i.e. the two arrays need to have the same length.

# Similar 2arg functions are defined in arraymath (not using
# vectorize_2arg).

# Using variable names: y=f(x1,x2) in gradient definitions.  The
# math2arg dictionary returns a pair for each function specifying
# expressions for first and second arg gradients.

# TODO: look at broadcast.jl
# Q: how to do max
# Q: where is +, .+, sum etc. some in arraymath.
# transpose, reshape, concat, copy?

math2arg = Dict{Symbol,Any}(
:+ => (1,1),
:.+ => (1,1),
:- => (1,-1),
:.- => (1,-1),
#:% => (1,-1), # this is wrong when x1<x2
#:.% => (1,-1),
:* => (:x2,:x1),
:.* => (:x2,:x1),
:/ => (:(1./x2),:(-x1./abs2(x2))),
:./ => (:(1./x2),:(-x1./abs2(x2))),
:\ => (:(-x2./abs2(x1)),:(1./x1)), # equivalent to x2/x1
:.\ => (:(-x2./abs2(x1)),:(1./x1)), # equivalent to x2/x1
:^ => (:(x2.*x1.^(x2-1)),:(y.*log(x1))),
:.^ => (:(x2.*x1.^(x2-1)),:(y.*log(x1))),
# :$ => bitwise xor
# :& => bitwise and
# :.<< => left bit shift
# :.>> => right bit shift
# :(|) => bitwise or
# # besselh
# # besseli
# # besselix
# # besselj
# # besseljx
# # besselk
# # besselkx
# # bessely
# # besselyx
# # beta
# # hankelh1
# # hankelh1x
# # hankelh2
# # hankelh2x
## :atan2
## :copysign
## :flipsign
## :hypot
## :log
# :max => (:(x1>=x2?1:0),:(x2>=x1?1:0)), # <= not implemented
# :min => (:(x1<=x2?1:0),:(x2<=x1?1:0)),
#minmax => returns a tuple, cannot multiply dy with .*
## airy
## airyx
## lbeta
## polygamma
## zeta
### :div
### :mod
### :rem
)

# typealias Aval{A,T} Union{T,A{T},Node{T},Node{A{T}}}
# typealias Bval{A,T} Union{Node{T},Node{A{T}}}
@primitive ^{T1<:Number,T2<:Integer}(x1::Node{T1},x2::T2) # to avoid clash with intfuncs:108

for (_f,_d) in math2arg
    @eval begin
        @primitive $_f{T1<:Number,T2<:Number}(x1::Node{T1},x2::Node{T2})
        @primitive $_f{T1<:Number,T2<:Number}(x1::Node{T1},x2::T2)
        @primitive $_f{T1<:Number,T2<:Number}(x1::T1,x2::Node{T2})
        $_f(::D1, y, x1, x2)=(dy->dy.*$(_d[1]))
        $_f(::D2, y, x1, x2)=(dy->dy.*$(_d[2]))
    end
end
