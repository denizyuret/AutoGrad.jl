# These are math functions in Julia that take a single argument.  They
# also handle array arguments where they apply element-wise.  This is
# implemented in Base using the @vectorize_1arg macro.  In fact I
# found these functions by grepping for @vectorize_1arg under base
# which led to the following files: complex, fastmath, floatfuncs,
# math, operators.  Also under special/ we have bessel, erf, gamma,
# log, trig.  I haven't tested these gradients for Complex numbers.
# TODO: There is also arraymath.jl and broadcast.jl to look at.

math1arg = Dict{Symbol,Any}(
:(+) => +1.0,  # supports (N,) (A,) (N,N) (N,A) (A,N) (A,A)
:(-) => -1.0,  # supports (N,) (A,) (N,N) (N,A) (A,N) (A,A)
#:(~) => :todo, # bitwise not, domain=Integer
#:(!) => :todo, # boolean not, domain=Bool
:abs2 => :(2.*x),
:abs => :(sign(x)),
:acos => :(-1./sqrt(1-abs2(x))),  # domain: abs(x) <= 1
:acosd => :(-(180/pi)./sqrt(1-abs2(x))),  # domain: abs(x) <= 1
:acosh => :(1./sqrt(abs2(x)-1)),  # domain: x >= 1
:acot => :(-1./(1+abs2(x))),
:acotd => :(-(180/pi)./(1+abs2(x))),
:acoth => :(1./(1-abs2(x))), # domain: abs(x) >= 1
:acsc => :(-1./sqrt(x.*x.*(x-1).*(x+1))), # domain: abs(x) >= 1
:acscd => :(-(180/pi)./sqrt(x.*x.*(x-1).*(x+1))), # domain: abs(x) >= 1
:acsch => :(-1./sqrt(x.^4+x.^2)),
:airy => :(airyprime(x)),   # airy(z)=airy(0,z); airy(k,x): The kth derivative of the Airy function Ai(x).
:airyai => :(airyprime(x)), # airyai(z)=airy(0,z)
#FAIL :airyaiprime => :(airybi(x)), # airyaiprime(z)=airy(1,z)
:airybi => :(airybiprime(x)), # airybi(z) = airy(2,z)
#FAIL :airybiprime => :(airy(4,x)), # airybiprime(z) = airy(3,z)
#FAIL :airyprime => :(airybi(x)),  # airyprime(z)=airy(1,z)
#:airyx => :todo, # airyx(z)=airyx(0,z)
#:angle => :todo, # angle(z::Real) = atan2(zero(z), z); angle(z::Complex) = atan2(imag(z), real(z))
:asec => :(1./sqrt(x.^4-x.^2)),  # domain: abs(x) >= 1
:asecd => :((180/pi)./sqrt(x.^4-x.^2)), # domain: abs(x) >= 1
:asech => :(-1./sqrt(x.^2-x.^4)), # domain: 0 < x <= 1
#:asin => :todo,   # domain: abs(x) <= 1
#:asind => :todo,  # domain: abs(x) <= 1
#:asinh => :todo,
#:atan => :todo,
#:atand => :todo,
#:atanh => :todo,
#:besselj0 => :todo,
#:besselj1 => :todo,
#:bessely0 => :todo,
#:bessely1 => :todo,
:cbrt => :(1./(3.*abs2(y))),
:ceil => 0,
#:cis => :todo,
#:conj => :todo,
#:cosc => :todo,
#:cosd => :todo,
#:cosh => :todo,
:cospi => :(-sinpi(x).*pi),
:cos => :(-sin(x)),
#:cot => :todo,
#:cotd => :todo,
#:coth => :todo,
#:csc => :todo,
#:cscd => :todo,
#:csch => :todo,
#:dawson => :todo,
:deg2rad => :(pi/180),
#:digamma => :todo,
#:erf => :todo,
#:erfc => :todo,
#:erfcinv => :todo,
#:erfcx => :todo,
#:erfi => :todo,
#:erfinv => :todo,
#:eta => :todo,
:exp10 => :(y.*log(10)),
:exp2 => :(y.*log(2)),
:expm1 => :(1+y),
:exponent => 0,
:exp => :y,
:float => 1,
:floor => 0,
#:gamma => :todo,
#:imag => :todo,
#:invdigamma => :todo,
:isfinite => 0,
:isinf => 0,
:isnan => 0,
#:lfact => :todo,
#:lgamma => :todo,
:log10 => :(1./(log(10).*x)),
:log1p => :(1./(1+x)),
:log => :(1./x),	# supports (N,) (A,) (N,N) (N,A) (A,N) (A,A)
:log2 => :(1./(log(2).*x)),
:rad2deg => :(180/pi),
#:real => :todo,
:round => 0,
#:sec => :todo,
#:secd => :todo,
#:sech => :todo,
:sign => 0,
#:significand => :todo,
#:sinc => :todo,
:sin => :(cos(x)),
#:sind => :todo,
#:sinh => :todo,
:sinpi => :(cospi(x).*pi),
:sqrt => :(1./(2.*y)),
:tan => :(1+abs2(y)),
#:tand => :todo,
:tanh => :(1-abs2(y)),
#:trigamma => :todo,
:trunc => 0,
#:zeta => :todo,
)

for (_f,_d) in math1arg
    if _d == 0
        @eval begin
            @zerograd $_f{T<:Number}(x::Node{T})
            # These are not allowed yet, see https://github.com/JuliaLang/julia/issues/3766
            # @zerograd $_f{T<:Number,A<:AbstractArray{T}}(x::Node{A})
            # @zerograd $_f{T<:Number,A<:AbstractArray}(x::Node{A{T}})
            @zerograd $_f{A<:AbstractArray}(x::Node{A})
        end
    else
        @eval begin
            @primitive $_f{T<:Number}(x::Node{T})
            @primitive $_f{A<:AbstractArray}(x::Node{A})
            $_f(::D1,y::Node,x)=(dy->dy.*$_d)
        end
    end
end

