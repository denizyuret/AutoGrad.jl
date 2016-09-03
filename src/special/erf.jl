erf1arg = [
(erf, :(exp(-abs2(x))*2/sqrt(pi)), (-Inf,Inf)),     # \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt
(erfc, :(-exp(-abs2(x))*2/sqrt(pi)), (-Inf,Inf)),   # 1-erf(x)
(erfcx, :(2y.*x-2/sqrt(pi)), (-Inf,Inf)),           # erfc(x)*exp(x^2)
(erfi, :(exp(abs2(x))*2/sqrt(pi)), (-Inf,Inf)),     # -i*erf(ix)
(dawson, :(-2y.*x+1), (-Inf,Inf)),                  # \frac{\sqrt{\pi}}{2} e^{-x^2} erfi(x).
(erfinv, :(exp(abs2(y))*sqrt(pi)/2), (-1,1)),       # erf(erfinv(x)) = x
(erfcinv, :(-exp(abs2(y))*sqrt(pi)/2), (0,2)),      # erfc(erfcinv(x)) = x
]

for (f,g,r) in erf1arg
    @eval @primitive $f(x),dy,y  (dy.*($g))
    if r==(-Inf,Inf)
        addtest(f,randn()); addtest(f,randn(2))
    elseif r==(-1,1)
        addtest(f,-1+2rand()); addtest(f,-1+2rand(2))
    elseif r==(0,2)
        addtest(f,2rand()); addtest(f,2rand(2))
    else
        error("Unknown range $r")
    end
end
