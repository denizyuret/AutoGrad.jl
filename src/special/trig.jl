trig1arg = [
(:acos, :(-1./sqrt_dot(1-abs2_dot(x))), (-1,1)),
(:acosd, :(-(180/pi)./sqrt_dot(1-abs2_dot(x))), (-1,1)),
(:acosh, :(1./sqrt_dot(abs2_dot(x)-1)), (1,Inf)),
(:acot, :(-1./(1+abs2_dot(x))), (-Inf,Inf)),
(:acotd, :(-(180/pi)./(1+abs2_dot(x))), (-Inf,Inf)),
(:acoth, :(1./(1-abs2_dot(x))), (-Inf,-1,1,Inf)),
(:acsc, :(-1./sqrt_dot(x.*x.*(x-1).*(x+1))), (-Inf,-1,1,Inf)),
(:acscd, :(-(180/pi)./sqrt_dot(x.*x.*(x-1).*(x+1))), (-Inf,-1,1,Inf)),
(:acsch, :(-1./sqrt_dot(x.^4+x.^2)), (-Inf,Inf)),
(:asec, :(1./sqrt_dot(x.^4-x.^2)), (-Inf,-1,1,Inf)),
(:asecd, :((180/pi)./sqrt_dot(x.^4-x.^2)), (-Inf,-1,1,Inf)),
(:asech, :(-1./sqrt_dot(x.^2-x.^4)), (0,1)),
(:asin, :(1./sqrt_dot(1-abs2_dot(x))), (-1,1)),
(:asind, :((180/pi)./sqrt_dot(1-abs2_dot(x))), (-1,1)),
(:asinh, :(1./sqrt_dot(1+abs2_dot(x))), (-Inf,Inf)),
(:atan, :(1./(1+abs2_dot(x))), (-Inf,Inf)),
(:atand, :((180/pi)./(1+abs2_dot(x))), (-Inf,Inf)),
(:atanh, :(1./(1-abs2_dot(x))), (-1,1)),
(:cos, :(-sin_dot(x)), (-Inf,Inf)),
# cos_kernel: Not exported
(:cosc, :(-2y./x-sinc_dot(x)*(pi^2)), (-Inf,Inf)),
(:cosd, :(-sind_dot(x)*pi/180), (-Inf,Inf)),
(:cosh, :(sinh_dot(x)), (-Inf,Inf)),
(:cospi, :(-sinpi_dot(x)*pi), (-Inf,Inf)),
(:cot, :(-abs2_dot(csc_dot(x))), (-Inf,Inf)),
(:cotd, :(-abs2_dot(cscd_dot(x))*pi/180), (-Inf,Inf)),
(:coth, :(-abs2_dot(csch_dot(x))), (-Inf,Inf)),
(:csc, :(-y.*cot_dot(x)), (-Inf,Inf)),
(:cscd, :(-y.*cotd_dot(x)*pi/180), (-Inf,Inf)),
(:csch, :(-y.*coth_dot(x)), (-Inf,Inf)),
# deg2rad_ext: Not exported
# mulpi_ext: Not exported
(:sec, :(y.*tan_dot(x)), (-Inf,Inf)),
(:secd, :(y.*tand_dot(x)*pi/180), (-Inf,Inf)),
(:sech, :(-y.*tanh_dot(x)), (-Inf,Inf)),
(:sin, :(cos_dot(x)), (-Inf,Inf)),
# sin_kernel: Not exported
(:sinc, :(cosc_dot(x)), (-Inf,Inf)), # sin(πx)/(πx)
(:sind, :(cosd_dot(x)*pi/180), (-Inf,Inf)),
(:sinh, :(cosh_dot(x)), (-Inf,Inf)),
(:sinpi, :(cospi_dot(x)*pi), (-Inf,Inf)),
(:tan, :(1+abs2_dot(y)), (-Inf,Inf)),
(:tand, :((1+abs2_dot(y))*pi/180), (-Inf,Inf)),
(:tanh, :(1-abs2_dot(y)), (-Inf,Inf)),
]

for (f,g,r) in trig1arg
    bf = broadcast_func(f)
    @eval @primitive $f(x),dy,y  (dy.*($g))
    if bf != f
        @eval @primitive $bf(x),dy,y  (dy.*($g))
    end
    addtest1(f,r)
end
