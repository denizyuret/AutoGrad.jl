# These are functions that can handle mixing scalar and array
# arguments.  Some of these functions come from grepping
# vectorize_2arg in Base (defined in operators, used in fastmath,
# floatfuncs, math, bessel, gamma), which allows them to have Array
# arguments in first, second, or both positions.  When both arguments
# are Arrays they must have the same size, or if one has extra
# dimensions at the end they need to be 1.  The resulting array will
# have the longer of the two sizes.  (implemented by promote_shape).
# Note that no broadcasting is performed here, i.e. the two arrays
# need to have the same length.

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
#:% => (1,:(-floor(x1./x2))), # does not work for (array,array)
:.% => (1,:(-floor(x1./x2))),
#:* => (:x2,:x1), # has a different (matmul) meaning for arrays
:.* => (:x2,:x1),
#:/ => (:(1./x2),:(-x1./abs2(x2))),  # has a different meaning when arg[2] is an array
:./ => (:(1./x2),:(-x1./abs2(x2))),
#:\ => (:(-x2./abs2(x1)),:(1./x1)), # equivalent to x2/x1, has a different meaning when arg[2] is an array
:.\ => (:(-x2./abs2(x1)),:(1./x1)), # equivalent to x2/x1
#:^ => (:(x2.*x1.^(x2-1)),:(y.*log(x1))), # does not work for array args
:.^ => (:(x2.*x1.^(x2-1)),:(y.*log(x1))),
#:(==) => 0,
:.== => 0,
#:< => 0,
:.< => 0,
#:<= => 0,
:.<= => 0,
#:> => 0,
:.> => 0,
#:>= => 0,
:.>= => 0,
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
:max => (:(y.==x1),:(y.==x2)), # <= not implemented
:min => (:(y.==x1),:(y.==x2)),
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
@primitive ^(x1::Node,x2::Integer) # to avoid clash with intfuncs:108

# Each argument can be Number, Array, Node{Number}, Node{Array}
# (unfortunately it is not currently possible to specify
# Node{Array{Number}}.  If at least one argument is a Node, we call
# the recorder method.

for (_f,_d) in math2arg
    if _d == 0
        @eval begin
            @zerograd $_f(x1::Node, x2::Node)
            @zerograd $_f(x1::Node, x2::Number)
            @zerograd $_f(x1::Node, x2::AbstractArray)
            @zerograd $_f(x1::Number,x2::Node)
            @zerograd $_f(x1::AbstractArray,x2::Node)
        end
    else
        @eval begin
            @primitive $_f(x1::Node, x2::Node)
            @primitive $_f(x1::Node, x2::Union{Number,AbstractArray})
            @primitive $_f(x1::Union{Number,AbstractArray},x2::Node)
            $_f(::D1, y, x1, x2)=(dbg(:dfxy, (name($_f),:D1,y,x1,x2));unbroadcast(y, x1, (dy->dy.*$(_d[1]))))
            $_f(::D2, y, x1, x2)=(dbg(:dfxy, (name($_f),:D1,y,x1,x2));unbroadcast(y, x2, (dy->dy.*$(_d[2]))))
        end
    end
end
