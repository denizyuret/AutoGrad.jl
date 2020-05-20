import SpecialFunctions: airy, airyai, airyaiprime, airyaiprimex, airyaix, airybi, airybiprime, airybiprimex, airybix, airyprime, airyx, besselh, besselhx, besseli, besselix, besselj, besselj0, besselj1, besseljx, besselk, besselkx, bessely, bessely0, bessely1, besselyx, beta, cosint, dawson, digamma, erf, erfc, erfcinv, erfcx, erfi, erfinv, eta, gamma, hankelh1, hankelh1x, hankelh2, hankelh2x, invdigamma, lbeta, lfact, lfactorial, polygamma, sinint, trigamma, zeta # , lgamma, logabsgamma, loggamma

# `airy(k,x)` is deprecated, use `airyai(x)`, `airyaiprime(x)`, `airybi(x)` or `airybiprime(x)` instead.
@primitive airyai(x),dy (dy.*(airyaiprime.(x)))
@primitive airyaiprime(x),dy (dy.*(x .* airyai.(x)))
# airyaiprimex
# airyaix
@primitive airybi(x),dy (dy.*(airybiprime.(x)))
@primitive airybiprime(x),dy (dy.*(x .* airybi.(x)))
# airybiprimex
# airybix
# `airyprime(z::Number)` is deprecated, use `airyaiprime(z)` instead.
# `airyx(k,x)` is deprecated, use `airyaix(x)`, `airyaiprimex(x)`, `airybix(x)` or `airybiprimex(x)` instead.
@primitive besselh(a,k,x),dy  nothing  nothing  (dy.*(besselh.(a-1,k,x) - besselh.(a+1,k,x))/2)
# besselhx
@primitive besseli(a,x),dy  nothing  (dy.*(besseli.(a-1,x) + besseli.(a+1,x))/2)
# besselix
@primitive besselj(a,x),dy  nothing  (dy.*(besselj.(a-1,x) - besselj.(a+1,x))/2)
@primitive besselj0(x),dy (dy.*(-(besselj1.(x))))
@primitive besselj1(x),dy (dy.*((besselj0.(x) - besselj.(2,x)) / 2))
# besseljx
@primitive besselk(a,x),dy  nothing  (dy.*(besselk.(a-1,x) + besselk.(a+1,x))/(-2))
# besselkx
@primitive bessely(a,x),dy  nothing  (dy.*(bessely.(a-1,x) - bessely.(a+1,x))/2)
@primitive bessely0(x),dy (dy.*(-(bessely1.(x))))
@primitive bessely1(x),dy (dy.*((bessely0.(x) - bessely.(2,x)) / 2))
# besselyx
# beta
# cosint
@primitive dawson(x),dy,y (dy.*((-2 .* y) .* x .+ 1))
@primitive digamma(x),dy,y (dy.*(trigamma.(x)))
@primitive erf(x),dy,y (dy.*(exp.(-(abs2.(x))) .* convert(eltype(x), 2 / √π)))
@primitive erfc(x),dy,y (dy.*(-(exp.(-(abs2.(x)))) .* convert(eltype(x), 2 / √π)))
@primitive erfcinv(x),dy,y (dy.*(-(exp.(abs2.(y))) .* convert(eltype(x), √π / 2)))
@primitive erfcx(x),dy,y (dy.*((2 .* y) .* x .- convert(eltype(x), 2 / √π)))
@primitive erfi(x),dy,y (dy.*(exp.(abs2.(x)) .* convert(eltype(x), 2 / √π)))
@primitive erfinv(x),dy,y (dy.*(exp.(abs2.(y)) .* convert(eltype(x), √π / 2)))
# eta
@primitive gamma(x),dy,y (dy.*(y .* digamma.(x)))
@primitive hankelh1(a,x),dy  nothing  (dy.*(hankelh1.(a-1,x) - hankelh1.(a+1,x))/2)
# hankelh1x
@primitive hankelh2(a,x),dy  nothing  (dy.*(hankelh2.(a-1,x) - hankelh2.(a+1,x))/2)
# hankelh2x
@primitive invdigamma(x),dy,y (dy.*(1 ./ trigamma.(y)))
# lbeta
# `lfact` is deprecated, use `lfactorial` instead.
# lfactorial
# logabsgamma
# `lgamma` is deprecated, use `(logabsgamma(x))[1]` instead. I use `loggamma` which throws a DomainError if gamma(x) is negative.
# `lgamma` deprecated, using loggamma.
# TODO: remove this once everybody uses loggamma and SpecialFunctions 0.8+
if !isdefined(SpecialFunctions, :loggamma) && isdefined(SpecialFunctions, :lgamma)
    import SpecialFunctions: lgamma
    loggamma(x) = lgamma(x)
end
if isdefined(SpecialFunctions, :loggamma) && !isdefined(SpecialFunctions, :lgamma)
    import SpecialFunctions: loggamma, logabsgamma
    lgamma(x) = loggamma(x)
end
@primitive lgamma(x),dy,y (dy.*(digamma.(x)))
@primitive loggamma(x),dy,y (dy.*(digamma.(x)))
@primitive polygamma(x1,x2),dy,y  nothing  unbroadcast(x2,dy.*polygamma(x1+1,x2))
# sinint
@primitive trigamma(x),dy,y (dy.*(polygamma.(2,x)))
# zeta
