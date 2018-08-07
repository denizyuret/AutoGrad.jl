import Base: acos, acosd, acosh, acot, acotd, acoth, acsc, acscd, acsch, asec, asecd, asech, asin, asind, asinh, atan, atand, atanh, cbrt, clamp, cos, cosc, cosd, cosh, cospi, cot, cotd, coth, csc, cscd, csch, deg2rad, exp, exp10, exp2, expm1, exponent, hypot, ldexp, log, log10, log1p, log2, mod2pi, rad2deg, sec, secd, sech, significand, sin, sinc, sind, sinh, sinpi, sqrt, tan, tand, tanh

# julia> for n in names(Base.Math); println(n); end
# @evalpoly
# Math
# ^
@primitive acos(x),dy (dy.*(-1 ./ sqrt.(1 .- abs2.(x))))
@primitive acosd(x),dy (dy.*(-(180 / pi) ./ sqrt.(1 .- abs2.(x))))
@primitive acosh(x),dy (dy.*(1 ./ sqrt.(abs2.(x) .- 1)))
@primitive acot(x),dy (dy.*(-1 ./ (1 .+ abs2.(x))))
@primitive acotd(x),dy (dy.*(-(180 / pi) ./ (1 .+ abs2.(x))))
@primitive acoth(x),dy (dy.*(1 ./ (1 .- abs2.(x))))
@primitive acsc(x),dy (dy.*(-1 ./ sqrt.((abs2.(x) .* (x .- 1)) .* (x .+ 1))))
@primitive acscd(x),dy (dy.*(-(180 / pi) ./ sqrt.((abs2.(x) .* (x .- 1)) .* (x .+ 1))))
@primitive acsch(x),dy (dy.*(-1 ./ sqrt.(abs2.(abs2.(x)) + abs2.(x))))
@primitive asec(x),dy (dy.*(1 ./ sqrt.(abs2.(abs2.(x)) - abs2.(x))))
@primitive asecd(x),dy (dy.*((180 / pi) ./ sqrt.(abs2.(abs2.(x)) - abs2.(x))))
@primitive asech(x),dy (dy.*(-1 ./ sqrt.(abs2.(x) - abs2.(abs2.(x)))))
@primitive asin(x),dy (dy.*(1 ./ sqrt.(1 .- abs2.(x))))
@primitive asind(x),dy (dy.*((180 / pi) ./ sqrt.(1 .- abs2.(x))))
@primitive asinh(x),dy (dy.*(1 ./ sqrt.(1 .+ abs2.(x))))
@primitive atan(x),dy (dy.*(1 ./ (1 .+ abs2.(x))))
@primitive atan(x1,x2),dy,y  unbroadcast(x1,dy.*(x2./(abs2.(x1)+abs2.(x2))))  unbroadcast(x2,dy.*(-x1./(abs2.(x1)+abs2.(x2))))
@primitive atand(x),dy (dy.*((180 / pi) ./ (1 .+ abs2.(x))))
@primitive atanh(x),dy (dy.*(1 ./ (1 .- abs2.(x))))
@primitive cbrt(x),dy,y (dy.*(1 ./ (3 .* abs2.(y))))
@primitive clamp(x,lo,hi),dy,y  unbroadcast(x,dy.*(lo .<= x .<= hi))
# clamp! overwrites.
@primitive cos(x),dy (dy.*(-(sin.(x))))
@primitive cosc(x),dy,y (dy.*((-2y) ./ x - sinc.(x) * pi ^ 2))
@primitive cosd(x),dy (dy.*((-(sind.(x)) * pi) / 180))
@primitive cosh(x),dy (dy.*(sinh.(x)))
@primitive cospi(x),dy (dy.*(-(sinpi.(x)) * pi))
@primitive cot(x),dy (dy.*(-(abs2.(csc.(x)))))
@primitive cotd(x),dy (dy.*((-(abs2.(cscd.(x))) * pi) / 180))
@primitive coth(x),dy (dy.*(-(abs2.(csch.(x)))))
@primitive csc(x),dy,y (dy.*(-y .* cot.(x)))
@primitive cscd(x),dy,y (dy.*(((-y .* cotd.(x)) * pi) / 180))
@primitive csch(x),dy,y (dy.*(-y .* coth.(x)))
@primitive deg2rad(x),dy (dy.*(pi / 180))
@primitive exp(x),dy,y (dy.*(y))
@primitive exp10(x),dy,y (dy.*(y .* log(10)))
@primitive exp2(x),dy,y (dy.*(y .* log(2)))
@primitive expm1(x),dy,y (dy.*(1 .+ y))
@zerograd exponent(x)
# frexp
@primitive hypot(x1,x2),dy,y  unbroadcast(x1,dy.*(x1./y))  unbroadcast(x2,dy.*(x2./y))
@primitive ldexp(x,n),dy  (dy.*(2 .^ n))
@primitive log(x),dy (dy.*(1 ./ x))
@primitive log(x1,x2),dy,y  unbroadcast(x1,dy.*(-log.(x2)./(x1.*abs2.(log.(x1)))))  unbroadcast(x2,dy.*(1 ./ (x2.*log.(x1))))
@primitive log10(x),dy (dy.*(1 ./ (log(10) .* x)))
@primitive log1p(x),dy (dy.*(1 ./ (1 .+ x)))
@primitive log2(x),dy (dy.*(1 ./ (log(2) .* x)))
# max
# min
# minmax
@primitive mod2pi(x),dy dy
# modf
@primitive rad2deg(x),dy (dy.*(180 / pi))
# rem2pi
@primitive sec(x),dy,y (dy.*(y .* tan.(x)))
@primitive secd(x),dy,y (dy.*(((y .* tand.(x)) * pi) / 180))
@primitive sech(x),dy,y (dy.*(-y .* tanh.(x)))
@primitive significand(x),dy (dy.*(2 .^ -exponent.(x)))
@primitive sin(x),dy (dy.*(cos.(x)))
@primitive sinc(x),dy (dy.*(cosc.(x)))
# sincos: TODO
@primitive sind(x),dy (dy.*((cosd.(x) * pi) / 180))
@primitive sinh(x),dy (dy.*(cosh.(x)))
@primitive sinpi(x),dy (dy.*(cospi.(x) * pi))
@primitive sqrt(x),dy,y (dy.*(1 ./ (2 .* y)))
@primitive tan(x),dy,y (dy.*(1 .+ abs2.(y)))
@primitive tand(x),dy,y (dy.*(((1 .+ abs2.(y)) * pi) / 180))
@primitive tanh(x),dy,y (dy.*(1 .- abs2.(y)))
