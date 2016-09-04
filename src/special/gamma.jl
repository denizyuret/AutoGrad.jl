gamma1arg = [
(:gamma, :(y.*digamma(x)), (-Inf,Inf)),
(:lfact, :(sign(y).*digamma(x+1)), (-Inf,Inf)),
(:lgamma, :(digamma(x)), (-Inf,Inf)),
(:digamma, :(trigamma(x)), (-Inf,Inf)), # polygamma(0,x)
(:trigamma, :(polygamma(2,x)), (-Inf,Inf)), # polygamma(1,x)
(:invdigamma, :(1./trigamma(y)), (-Inf,Inf)),
# zeta: TODO. Riemann 1-arg zeta
# eta # TODO. related to zeta
]

for (f,g,r) in gamma1arg
    @eval @primitive $f(x),dy,y  (dy.*($g))
    addtest1(f,r)
end

gamma2arg = [
#:beta => :TODO,                          # gamma,operators
#:lbeta => :TODO,                         # gamma,operators
#:zeta => :TODO,                 # Hurwitz 2-arg zeta
]

# polygamma wants x1 to be a non-negative integer, x2 unrestricted
@primitive polygamma(x1,x2),dy,y  nothing  unbroadcast(x2,dy.*polygamma(x1+1,x2))
addtest2(polygamma, 0:5, (-Inf,Inf))
