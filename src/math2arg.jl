# math2arg: These are functions that can handle mixing scalar and array
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
# expressions to compute the first and second arg gradients.

# Unmarked functions below support the default signatures:
# (N,N) (N,A) (A,N) (A,A)
# Extra or missing argtypes are marked.
# where N:number, A:array, (A,B) different sized arrays.

math2arg = Dict{Symbol,Any}(
:+ => (1,1),                     # extra (N,) (A,)
:- => (1,-1),                    # extra (N,) (A,)
:^ => (:(x2.*x1.^(x2-1)),:(y.*log(x1))), # only supports (N,N), arrays not supported
:rem => (1,:(-trunc(x1./x2))),   # Remainder from Euclidean division, return same sign as x, missing (A,A)
# :% => (1,:(-trunc(x1./x2))),   # same as rem
:mod => (1,:(-floor(x1./x2))),   # Modulus after division, return same sign as y.
#:airy => :todo,                  # first arg should be an integer
#:airyx => :todo,                 # first arg should be an integer
#:atan2 => :todo,
#:besselh => :todo,
#:besseli => :todo,
#:besselix => :todo,
#:besselj => :todo,
#:besseljx => :todo,
#:besselk => :todo,
#:besselkx => :todo,
#:bessely => :todo,
#:besselyx => :todo,
#:beta => :todo,
#:copysign => :todo,
#:flipsign => :todo,
#:hankelh1 => :todo,
#:hankelh1x => :todo,
#:hankelh2 => :todo,
#:hankelh2x => :todo,
#:hypot => :todo,
#:lbeta => :todo,
#:log => :todo,                  # extra (N,) (A,)
:max => (:(y.==x1),:(y.==x2)),
:min => (:(y.==x1),:(y.==x2)),
#:minmax => :todo,               # only supports (N,N); returns a tuple, cannot multiply dy with .*
#:polygamma => :todo,            # first argument should be an integer
#:zeta => :todo,                 # domain >= 1?
)

# Each argument can be Number, Array, Node{Number}, Node{Array}
# (unfortunately it is not currently possible to specify
# Node{Array{Number}}.  If at least one argument is a Node, we call
# the recorder method.

@primitive ^(x1::Node,x2::Integer) # to avoid clash with intfuncs:108
for (_f,_d) in math2arg
    @eval begin
        @primitive $_f(x1::Node, x2::Node)
        @primitive $_f(x1::Node, x2::Union{Number,AbstractArray})
        @primitive $_f(x1::Union{Number,AbstractArray},x2::Node)
        $_f(::D1, y, x1, x2)=unbroadcast(y, x1, (dy->dy.*$(_d[1])))
        $_f(::D2, y, x1, x2)=unbroadcast(y, x2, (dy->dy.*$(_d[2])))
    end
end

# math2zerograd:
# These functions have zero gradient either because they return boolean 
# values or they truncate to an integer:

math2zerograd = Dict{Symbol,Any}(
:.<  => 0,
:.<= => 0,
:.== => 0,
:.>  => 0,
:.>= => 0,
:div => 0,                       # The quotient from Euclidean division. Computes x/y, truncated to an integer.
# :÷ => 0, 			 # Same as div.
:(==) => 0,                      # supports any pair of values
:< => 0,                         # only supports (N,N), arrays not supported
:<= => 0,                        # only supports (N,N), arrays not supported
:> => 0,                         # only supports (N,N), arrays not supported
:>= => 0,                        # only supports (N,N), arrays not supported
#:$ => :todo,                     # domain: Integers, bitwise xor
#:& => :todo,                     # domain: Integers, bitwise and
#:| => :todo,                     # domain: Integers, bitwise or
#:.<< => :todo,                   # domain: Integers, left bit shift
#:.>> => :todo,                   # domain: Integers, right bit shift
)

for (_f,_d) in math2zerograd
    @eval begin
        @zerograd $_f(x1::Node, x2::Node)
        @zerograd $_f(x1::Node, x2::Number)
        @zerograd $_f(x1::Node, x2::AbstractArray)
        @zerograd $_f(x1::Number,x2::Node)
        @zerograd $_f(x1::AbstractArray,x2::Node)
    end
end

# math2broadcast: 
# These functions use broadcasting to handle arrays of different sizes.
# Unless otherwise specified they support:
# (N,N) (N,A) (A,N) (A,A) (A,B)
# where N:Number, A,B arrays of broadcast compatible sizes.

math2broadcast = Dict{Symbol,Any}(
:.+ => (1,1),                    # extra (A,)
:.* => (:x2,:x1),                # extra (A,)
:.- => (1,-1),
:.% => (1,:(-trunc(x1./x2))),
:./ => (:(1./x2),:(-x1./abs2(x2))),
:.\ => (:(-x2./abs2(x1)),:(1./x1)),
:.^ => (:(x2.*x1.^(x2-1)),:(y.*log(x1))), # domain: x1 >= 0 (unless we use complex args)
)

for (_f,_d) in math2broadcast
    @eval begin
        @primitive $_f(x1::Node, x2::Node)
        @primitive $_f(x1::Node, x2::Union{Number,AbstractArray})
        @primitive $_f(x1::Union{Number,AbstractArray},x2::Node)
        $_f(::D1, y, x1, x2)=unbroadcast(y, x1, (dy->dy.*$(_d[1])))
        $_f(::D2, y, x1, x2)=unbroadcast(y, x2, (dy->dy.*$(_d[2])))
    end
end

# These operators do not broadcast, and they have special meanings for array arguments.
# Type legend: N=Number, V=Vector, M=Matrix, T=Tensor (ndims>2)

math2linalg = Dict{Symbol,Any}(
:* => (:x2,:x1),                   # (N,) (M,) (N,*) (*,N) (M,V) (M,M)
:/ => (:(1./x2),:(-x1./abs2(x2))), # (*,N) (V,V) (M,M)
:\ => (:(-x2./abs2(x1)),:(1./x1)), # (N,*) (V,V) (M,M) (M,V) (V,M)
)


# TODO: look at broadcast.jl
# TODO: transpose, reshape, concat, copy?
# TODO: anything else in linalg?
