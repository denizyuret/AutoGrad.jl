# These are functions defined in base/math.jl.  They apply elementwise
# to array arguments.  This is implemented in operators.jl using the
# @vectorize_1arg macro.  The first element of the tuples below gives
# the function name.  The second element is an expression for dy/dx
# where y=f(x).  The gradient expressions should be valid for array
# inputs.  The last part gives the range for generating test cases.

math1arg = [
(:cbrt, :(1./(3.*abs2.(y))), (-Inf,Inf)),
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
(:significand, :(0.5.^exponent.(x)), (-Inf,Inf)),
(:sqrt, :(1./(2.*y)), (0,Inf)),
]

for (f,g,r) in math1arg
    bf = broadcast_func(f)
    @eval @primitive $f(x),dy,y  (dy.*($g))
    if f != bf
        @eval @primitive $bf(x),dy,y  (dy.*($g))
    end
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
(:atan2, quote x2./(abs2.(x1)+abs2.(x2)) end, quote -x1./(abs2.(x1)+abs2.(x2)) end),
(:hypot, quote x1./y end, quote x2./y end),
(:max, quote y.==x1 end, quote y.==x2 end),
(:min, quote y.==x1 end, quote y.==x2 end),
(:log, quote -log.(x2)./(x1.*abs2.(log.(x1))) end, quote 1./(x2.*log.(x1)) end),
]

# The 2-arg log supports positive args for reals.
log(x1::Irrational{:e},x2::Rec)=log(float(x1),x2) # to avoid clash with irrationals.jl:131.

for (f,g1,g2) in math2arg
    bf = broadcast_func(f)
    @eval @primitive $f(x1,x2),dy,y  unbroadcast(x1,dy.*($g1))  unbroadcast(x2,dy.*($g2))
    if f != bf
        @eval @primitive $bf(x1,x2),dy,y  unbroadcast(x1,dy.*($g1))  unbroadcast(x2,dy.*($g2))
    end
    addtest2(f, (f==:log ? (0,Inf) : (-Inf,Inf)))
end

# ^ only supports (N>=0,N), arrays not supported in math.jl, only M^N in linalg/dense.jl (TODO)
(^){T<:AbstractFloat}(x1::Rec{T},x2::Integer)=(^)(x1,convert(eltype(x1.value),x2)) # to avoid clash with intfuncs:199
(^)(x1::Broadcasted,x2::Integer)=(^)(x1,convert(eltype(x1.value),x2)) # to avoid clash with intfuncs:199
@primitive (^)(x1::Number,x2::Number),dy,y  (dy*x2*x1^(x2-1))  (dy*y*log.(x1))
addtestN(:^, randin((0,Inf)), randin((-Inf,Inf)))

@primitive clamp(x,lo,hi),dy,y  unbroadcast(x,dy.*(lo .<= x .<= hi))
bf = broadcast_func(:clamp)
if bf != :clamp
    @eval @primitive $bf(x,d...),dy,y  unbroadcast(x,dy.*(d[1] .<= x .<= d[2]))
end
addtest(:clamp, randn(), -1., 1.)
addtest(bf, randn(10), -1., 1.)
broadcast_func(:&)  # need this for (lo .<= x .<= hi)

# ldexp(x,n) computes x*2^n with x real, n integer
@primitive ldexp(x,n...),dy  (dy*(2.0^n[1]))
addtest(:ldexp, randn(), rand(-2:2))
bf = broadcast_func(:ldexp)
if bf != :ldexp
    @eval @primitive $bf(x,n...),dy  (dy.*(2.0^n[1]))    
    addtest(bf, randn(2), rand(-2:2))
end

# mod2pi(x) returns modulus after division by 2pi for x real.
@primitive mod2pi(x),dy dy
addtest(:mod2pi, 100randn())
bf = broadcast_func(:mod2pi)
if bf != :mod2pi
    @eval @primitive $bf(x),dy dy
    addtest(bf, 100randn(2))
end

# zerograd functions
bf = broadcast_func(:exponent)
@zerograd exponent(x)
if bf != :exponent
    @eval @zerograd $bf(x)
end

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
