# These are functions defined in math.jl.  They can handle array arguments where
# they apply element-wise.  This is implemented in operators.jl using the
# @vectorize_1arg macro.  The values in the math1arg hash are expressions for
# dy/dx where y=f(x) and x,y can be arrays or scalars.  These expressions should
# be valid for array inputs.

math1arg = Dict{Symbol,Any}(
:acos => :(-1./sqrt(1-abs2(x))),	# domain: abs(x) <= 1; math,operators
:acosh => :(1./sqrt(abs2(x)-1)),        # domain: x >= 1; math,operators
:asin => :(1./sqrt(1-abs2(x))),         # domain: abs(x) <= 1; math,operators
:asinh => :(1./sqrt(1+abs2(x))),        # math,operators
:atan => :(1./(1+abs2(x))),             # math,operators
:atanh => :(1./(1-abs2(x))),            # math,operators
:cbrt => :(1./(3.*abs2(y))),            # math,operators
:cos => :(-sin(x)),                     # math,operators
:cosh => :(sinh(x)),                    # math,operators
:deg2rad => :(pi/180),                  # math,operators
:erf => :(2exp(-abs2(x))/sqrt(pi)),     # math,operators
:erfc => :(-2exp(-abs2(x))/sqrt(pi)),   # math,operators
:exp => :y,                             # math,operators
:exp10 => :(y.*log(10)),                # math,operators
:exp2 => :(y.*log(2)),                  # math,operators
:expm1 => :(1+y),                       # math,operators
:exponent => 0,                         # returns int; math,operators
:lgamma => :(digamma(x)),               # math,operators
:log => :(1./x),                        # supports (N,) (A,) (N,N) (N,A) (A,N) (A,A); math,operators
:log10 => :(1./(log(10).*x)),           # math,operators
:log1p => :(1./(1+x)),                  # math,operators
:log2 => :(1./(log(2).*x)),             # math,operators
:rad2deg => :(180/pi),                  # math,operators
:significand => :(0.5.^exponent(x)),    # math,operators
:sin => :(cos(x)),                      # math,operators
:sinh => :(cosh(x)),                    # math,operators
:sqrt => :(1./(2.*y)),                  # math,operators
:tan => :(1+abs2(y)),                   # math,operators
:tanh => :(1-abs2(y)),                  # math,operators
)

for (f,g) in math1arg
    @eval @primitive  $f(x::AorN)::y  (dy->dy.*$g)
    addtest(f, randn())
    addtest(f, randn(2))
end

for f in (:log, :log2, :log10, :sqrt); fixdomain(::Fn{f},x)=(abs(x),); end
for f in (:acos, :asin, :atanh); fixdomain(::Fn{f},x)=(sin(x),); end
fixdomain(::Fn{:log1p},x)=(abs(x)-1,)
fixdomain(::Fn{:acosh},x)=(abs(x)+1,)


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
:atan2 => (:(x2./(abs2(x1)+abs2(x2))), :(-x1./(abs2(x1)+abs2(x2)))), # math,operators
:hypot => (:(x1./y),:(x2./y)),    # math,operators
:log => (:(-log(x2)./(x1.*abs2(log(x1)))),:(1./(x2.*log(x1)))), # extra (N,) (A,); math,operators
:max => (:(y.==x1),:(y.==x2)),    # math,operators; 
:min => (:(y.==x1),:(y.==x2)),    # math,operators; 
)

log{T<:AorN}(x1::Irrational{:e},x2::Node{T})=log(float(x1),x2) # to avoid clash with irrationals.jl:131.

for (f,g) in math2arg
    @eval @primitive $f(x1::AorN,x2::AorN)::y  unbroadcast(y,x1,dy->dy.*$(g[1]))  unbroadcast(y,x2,dy->dy.*$(g[2]))
    addtest(f, randn(), randn())
    addtest(f, randn(), randn(2))
    addtest(f, randn(2), randn())
    addtest(f, randn(2), randn(2))
end

fixdomain(::Fn{:log},x,y)=(abs(x),abs(y))

# @primitive log{T<:Number}(x1::Base.Irrational{:e}, x2::Node{T}) # to avoid clash with irrationals.jl:131.
# @primitive log{T<:AbstractArray}(x1::Base.Irrational{:e}, x2::Node{T}) # to avoid clash with irrationals.jl:131.
# defgrads(math2arg, Number, Number)
# defgrads(math2arg, AbstractArray, Number)
# defgrads(math2arg, Number, AbstractArray)
# defgrads(math2arg, AbstractArray, AbstractArray)

# math2arg1 = Dict{Symbol,Any}(
# :^ => (:(x2.*x1.^(x2-1)),:(y.*log(x1))), # only supports (N,N), arrays not supported in math.jl, only M^N in linalg/dense.jl
# #:minmax => :todo, # only supports (N,N); returns a tuple, cannot multiply dy with .*; math,operators
# )                             

# @primitive ^{T<:Number}(x1::Node{T},x2::Integer) # to avoid clash with intfuncs:108
# defgrads(math2arg1, Number, Number)
# testargs(::Fn{:^},x...)=map(abs,testargs(Fn2(:^),x...))

(^){T<:Number}(x1::Node{T},x2::Integer)=(^)(x1,float(x2)) # to avoid clash with intfuncs:108
@primitive (^)(x1::Number,x2::Number)::y  (dy->dy*x2*x1^(x2-1))  (dy->dy*y*log(x1))
addtest(:(^), randn(), randn())
fixdomain(::Fn{:^},x,y)=(abs(x),y)

# TODO:

# eval
# clamp
# clamp!
# rad2deg
# deg2rad
# log
# $(Expr(:$, :f)): Not a symbol
# cbrt
# exp2
# exp10
# sqrt
# hypot
# atan2
# max
# min
# minmax
# ldexp
# exponent
# significand
# frexp
# modf
# ^
# angle_restrict_symm: Not exported
# add22condh: Not exported
# ieee754_rem_pio2: Not exported
# mod2pi


### DEAD CODE:
# defgrads(math1arg, Number)
# defgrads(math1arg, AbstractArray)

# for (f,g) in ((:acos, :cos),
#               (:acosh, :cosh),
#               (:asin, :sin),
#               (:asinh, :sinh),
#               (:atan, :tan),
#               (:atanh, :tanh))
#     gx = eval(g)
#     testargs(::Fn{f},a...)=map(gx, testargs(Fn2(f),a...))
# end
# for f in (:log1p,)
#     testargs(::Fn{f},a...)=map(x->-1+abs(x), testargs(Fn2(f),a...))
# end
# for f in (:log, :log2, :log10, :sqrt)
#     testargs(::Fn{f},a...)=map(abs, testargs(Fn2(f),a...))
# end
