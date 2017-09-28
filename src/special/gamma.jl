gamma1arg = [
(:gamma, :(y.*digamma_dot(x)), (-Inf,Inf)),
# (:lfact, :(sign_dot(y).*digamma_dot(x+1)), (-Inf,Inf)), # lfact only defined for integers
(:lgamma, :(digamma_dot(x)), (-Inf,Inf)),
(:digamma, :(trigamma_dot(x)), (-Inf,Inf)), # polygamma(0,x)
(:trigamma, :(polygamma2_dot(x)), (-Inf,Inf)), # polygamma(1,x)
(:invdigamma, :(1./trigamma_dot(y)), (-Inf,Inf)),
(:polygamma2, :(error()), (-Inf,Inf)),
# zeta: TODO. Riemann 1-arg zeta
# eta # TODO. related to zeta
]

polygamma2(x)=polygamma(2,x)
if VERSION >= v"0.6.0"
    polygamma2_dot(x)=polygamma2.(x)
else
    polygamma2_dot(x)=polygamma2(x)
end

for (f,g,r) in gamma1arg
    bf = broadcast_func(f)
    @eval @primitive $f(x),dy,y  (dy.*($g))
    if bf != f
        @eval @primitive $bf(x),dy,y  (dy.*($g))
    end
    if f != :polygamma2
        addtest1(f,r)
    end
end

gamma2arg = [
#:beta => :TODO,                          # gamma,operators
#:lbeta => :TODO,                         # gamma,operators
#:zeta => :TODO,                 # Hurwitz 2-arg zeta
]

# polygamma wants x1 to be a non-negative integer, x2 unrestricted
@primitive polygamma(x1,x2),dy,y  nothing  unbroadcast(x2,dy.*polygamma(x1+1,x2))
polygamma_(x,i)=polygamma(i,x)
addtest(:polygamma_, randn(), rand(0:5))
# TODO: add broadcasting version

