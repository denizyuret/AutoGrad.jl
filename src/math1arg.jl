# These are math functions in Julia that take a single argument.  They
# also handle array arguments where they apply element-wise.  This is
# implemented in Base using the @vectorize_1arg macro.  In fact I
# found these functions by grepping for @vectorize_1arg under base
# which led to the following files: complex, fastmath, floatfuncs,
# math, operators.  Under special/ we have bessel, erf, gamma, log,
# trig.  I haven't tested these gradients for Complex numbers.  TODO:
# There is also arraymath.jl and broadcast.jl to look at.

math1arg = Dict{Symbol,Any}(
:(+) => +1.0,
:(-) => -1.0,
# :(~)
# :(!)
:abs2 => :(2x),
:abs => :(sign(x)),
:acos => :(-1./sqrt(1-abs2(x))),
:acosd => :(-(180/pi)./sqrt(1-abs2(x))),
# acosh => :(1./sqrt(abs2(x)-1)),
:acot => :(-1./(1+abs2(x))),
:acotd => :(-(180/pi)./(1+abs2(x))),
# acoth => :(1./(1-abs2(x))),
# acsc
# acscd
# acsch
# airy
# airyai
# airyaiprime
# airybi
# airybiprime
# airyprime
# airyx
# angle
# asec
# asecd
# asech
# asin
# asind
# asinh
# atan
# atand
# atanh
# besselj0
# besselj1
# bessely0
# bessely1
:cbrt => :(1./(3*abs2(y))),
# ceil
# cis
# conj
# cosc
# cosd
# cosh
# :cospi => :(-sinpi(x)*pi),
:cos => :(-sin(x)),
# cot
# cotd
# coth
# csc
# cscd
# csch
# Dawson
:deg2rad => :(pi/180),
# digamma
# erf
# erfc
# erfcinv
# erfcx
# erfi
# erfinv
# eta
:exp10 => :(y*log(10)),
:exp2 => :(y*log(2)),
:expm1 => :(1+y),
# exponent
:exp => :y,
# float
# floor
# gamma
# imag
# invdigamma
# isfinite
# isinf
# isnan
# lfact
# lgamma
:log10 => :(1./(log(10)*x)),
:log1p => :(1./(1+x)),
:log => :(1./x),
:log2 => :(1./(log(2)*x)),
:rad2deg => :(180/pi),
# real
# round
# sec
# secd
# sech
:sign => 0,
# significand
# sinc
:sin => :(cos(x)),
# sind
# sinh
# :sinpi => :(cospi(x)*pi),
:sqrt => :(1./2y),
:tan => :(1+abs2(y)),
# tand
:tanh => :(1-abs2(y)),
# trigamma
# trunc
# zeta
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
            $_f(::D1,y,x)=(dy->dy.*$_d)
        end
    end
end

