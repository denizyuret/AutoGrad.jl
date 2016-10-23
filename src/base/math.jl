# These are functions defined in base/math.jl.  They apply elementwise
# to array arguments.  This is implemented in operators.jl using the
# @vectorize_1arg macro.  The first element of the tuples below gives
# the function name.  The second element is an expression for dy/dx
# where y=f(x).  The gradient expressions should be valid for array
# inputs.  The last part gives the range for generating test cases.

math1arg = [
(:cbrt, :(1./(3.*abs2(y))), (-Inf,Inf)),
(:deg2rad, :(pi/180), (-Inf,Inf)),
(:exp, :y, (-Inf,Inf)),
(:exp10, :(y.*log(10)), (-Inf,Inf)),
(:exp2, :(y.*log(2)), (-Inf,Inf)),
(:expm1, :(1+y), (-Inf,Inf)),
(:log, :(1./x), (0,Inf)),
(:log10, :(1./(log(10).*x)), (0,Inf)),
(:log1p, :(1./(1+x)), (-1,Inf)),
(:log2, :(1./(log(2).*x)), (0,Inf)),
(:rad2deg, :(180/pi), (-Inf,Inf)),
(:significand, :(0.5.^exponent(x)), (-Inf,Inf)),
(:sqrt, :(1./(2.*y)), (0,Inf)),
]

for (f,g,r) in math1arg
    @eval @primitive $f(x),dy,y  (dy.*($g))
    addtest1(f,r)
end


# math2arg: These are two argument functions that can handle mixing
# scalar and array arguments.  They use vectorize_2arg in Base
# (defined in operators) which allows them to have Array arguments in
# first, second, or both positions.  When both arguments are Arrays
# they must have the same size, or if one has extra dimensions at the
# end, they need to be 1.  The resulting array will have the longer of
# the two sizes.  (implemented by promote_shape).  Note that no
# broadcasting is performed here, i.e. the two arrays need to have the
# same length.  Note that some 2arg functions are defined in arraymath
# not using vectorize_2arg.  Using variable names: y=f(x1,x2) in
# gradient definitions.

math2arg = [
(:atan2, :(x2./(abs2(x1)+abs2(x2))), :(-x1./(abs2(x1)+abs2(x2)))),
(:hypot, :(x1./y), :(x2./y)),
(:max, :(y.==x1), :(y.==x2)),
(:min, :(y.==x1), :(y.==x2)),
]

for (f,g1,g2) in math2arg
    @eval @primitive $f(x1,x2),dy,y  unbroadcast(x1,dy.*($g1))  unbroadcast(x2,dy.*($g2))
    addtest2(f,(-Inf,Inf))
end

# The 2-arg log supports positive args for reals.
log(x1::Irrational{:e},x2::Rec)=log(float(x1),x2) # to avoid clash with irrationals.jl:131.
@primitive log(x1,x2),dy  unbroadcast(x1,-dy.*log(x2)./(x1.*abs2(log(x1))))  unbroadcast(x2,dy./(x2.*log(x1)))
addtest2(log,(0,Inf))

# ^ only supports (N>=0,N), arrays not supported in math.jl, only M^N in linalg/dense.jl (TODO)
(^){T<:Number}(x1::Rec{T},x2::Integer)=(^)(x1,float(x2)) # to avoid clash with intfuncs:108
@primitive (^)(x1::Number,x2::Number),dy,y  (dy*x2*x1^(x2-1))  (dy*y*log(x1))
addtest(^, randin((0,Inf)), randin((-Inf,Inf)))

# clamp(x,lo,hi) clamps x between lo and hi
@primitive clamp(x,i...),dy,y  unbroadcast(x,dy.*(i[1] .<= x .<= i[2]))
addtest(clamp, randn(10), -1., 1.)
addtest(clamp, randn(), -1., 1.)

# ldexp(x,n) computes x*2^n with x real, n integer
@primitive ldexp(x,n...),dy  (dy*(2.0^n[1]))
addtest(ldexp, randn(), rand(-2:2))

# mod2pi(x) returns modulus after division by 2pi for x real.
@primitive mod2pi(x::Number),dy dy
addtest(mod2pi, 100randn())

# zerograd functions
@zerograd exponent(x)


# Other functions defined in julia/base/math.jl
# add22condh: Not exported
# angle_restrict_symm: Not exported
# clamp!: overwriting
# frexp: returns tuple
# ieee754_rem_pio2: Not exported
# minmax: returns tuple
# modf: returns tuple
# Moved to erf.jl:
# erf: see erf.jl
# erfc: see erf.jl
# The following moved to trig.jl:
# (acos, :(-1./sqrt(1-abs2(x))), (-1,1)),
# (acosh, :(1./sqrt(abs2(x)-1)), (1,Inf)),
# (asin, :(1./sqrt(1-abs2(x))), (-1,1)),
# (asinh, :(1./sqrt(1+abs2(x))), (-Inf,Inf)),
# (atan, :(1./(1+abs2(x))), (-Inf,Inf)),
# (atanh, :(1./(1-abs2(x))), (-1,1)),
# (cos, :(-sin(x)), (-Inf,Inf)),
# (cosh, :(sinh(x)), (-Inf,Inf)),
# (sin, :(cos(x)), (-Inf,Inf)),
# (sinh, :(cosh(x)), (-Inf,Inf)),
# (tan, :(1+abs2(y)), (-Inf,Inf)),
# (tanh, :(1-abs2(y)), (-Inf,Inf)),
# Moved to gamma.jl:
# (lgamma, :(digamma(x)), (-Inf,Inf)),
