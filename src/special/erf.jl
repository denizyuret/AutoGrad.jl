erf1arg = [
(erf, :(exp(-abs2(x))*2/sqrt(pi))),     # \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt
(erfc, :(-exp(-abs2(x))*2/sqrt(pi))),   # 1-erf(x)
(erfcx, :(2y.*x-2/sqrt(pi))),           # erfc(x)*exp(x^2)
(erfi, :(exp(abs2(x))*2/sqrt(pi))),     # -i*erf(ix)
(dawson, :(-2y.*x+1)),                  # \frac{\sqrt{\pi}}{2} e^{-x^2} erfi(x).
(erfinv, :(exp(abs2(y))*sqrt(pi)/2)),   # erf(erfinv(x)) = x
(erfcinv, :(-exp(abs2(y))*sqrt(pi)/2)), # erfc(erfcinv(x)) = x
]

for (f,g) in erf1arg
    @eval @primitive $f(x),dy,y  (dy.*($g))
    addtest(f,randn()); addtest(f,randn(2))
end
