trig1arg = [
(:acos, :(-1./sqrt(1-abs2(x))), (-1,1)),
(:acosd, :(-(180/pi)./sqrt(1-abs2(x))), (-1,1)),
(:acosh, :(1./sqrt(abs2(x)-1)), (1,Inf)),
(:acot, :(-1./(1+abs2(x))), (-Inf,Inf)),
(:acotd, :(-(180/pi)./(1+abs2(x))), (-Inf,Inf)),
(:acoth, :(1./(1-abs2(x))), (-Inf,-1,1,Inf)),
(:acsc, :(-1./sqrt(x.*x.*(x-1).*(x+1))), (-Inf,-1,1,Inf)),
(:acscd, :(-(180/pi)./sqrt(x.*x.*(x-1).*(x+1))), (-Inf,-1,1,Inf)),
(:acsch, :(-1./sqrt(x.^4+x.^2)), (-Inf,Inf)),
(:asec, :(1./sqrt(x.^4-x.^2)), (-Inf,-1,1,Inf)),
(:asecd, :((180/pi)./sqrt(x.^4-x.^2)), (-Inf,-1,1,Inf)),
(:asech, :(-1./sqrt(x.^2-x.^4)), (0,1)),
(:asin, :(1./sqrt(1-abs2(x))), (-1,1)),
(:asind, :((180/pi)./sqrt(1-abs2(x))), (-1,1)),
(:asinh, :(1./sqrt(1+abs2(x))), (-Inf,Inf)),
(:atan, :(1./(1+abs2(x))), (-Inf,Inf)),
(:atand, :((180/pi)./(1+abs2(x))), (-Inf,Inf)),
(:atanh, :(1./(1-abs2(x))), (-1,1)),
(:cos, :(-sin(x)), (-Inf,Inf)),
# cos_kernel: Not exported
(:cosc, :(-2y./x-sinc(x)*(pi^2)), (-Inf,Inf)),
(:cosd, :(-sind(x)*pi/180), (-Inf,Inf)),
(:cosh, :(sinh(x)), (-Inf,Inf)),
(:cospi, :(-sinpi(x).*pi), (-Inf,Inf)),
(:cot, :(-abs2(csc(x))), (-Inf,Inf)),
(:cotd, :(-abs2(cscd(x))*pi/180), (-Inf,Inf)),
(:coth, :(-abs2(csch(x))), (-Inf,Inf)),
(:csc, :(-y.*cot(x)), (-Inf,Inf)),
(:cscd, :(-y.*cotd(x)*pi/180), (-Inf,Inf)),
(:csch, :(-y.*coth(x)), (-Inf,Inf)),
# deg2rad_ext: Not exported
# mulpi_ext: Not exported
(:sec, :(y.*tan(x)), (-Inf,Inf)),
(:secd, :(y.*tand(x)*pi/180), (-Inf,Inf)),
(:sech, :(-y.*tanh(x)), (-Inf,Inf)),
(:sin, :(cos(x)), (-Inf,Inf)),
# sin_kernel: Not exported
(:sinc, :(cosc(x)), (-Inf,Inf)), # sin(πx)/(πx)
(:sind, :(cosd(x)*pi/180), (-Inf,Inf)),
(:sinh, :(cosh(x)), (-Inf,Inf)),
(:sinpi, :(cospi(x).*pi), (-Inf,Inf)),
(:tan, :(1+abs2(y)), (-Inf,Inf)),
(:tand, :((1+abs2(y))*pi/180), (-Inf,Inf)),
(:tanh, :(1-abs2(y)), (-Inf,Inf)),
]

for (f,g,r) in trig1arg
    @eval @primitive $f(x),dy,y  (dy.*($g))
    addtest1(f,r)
end
