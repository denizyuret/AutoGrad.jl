gamma1arg = [
(:gamma, :(y.*digamma.(x)), (-Inf,Inf)),
# (:lfact, :(sign.(y).*digamma.(x+1)), (-Inf,Inf)), # lfact only defined for integers
(:lgamma, :(digamma.(x)), (-Inf,Inf)),
(:digamma, :(trigamma.(x)), (-Inf,Inf)), # polygamma(0,x)
(:trigamma, :(polygamma2.(x)), (-Inf,Inf)), # polygamma(1,x)
(:invdigamma, :(1./trigamma.(y)), (-Inf,Inf)),
(:polygamma2, :(error()), (-Inf,Inf)),
# zeta: TODO. Riemann 1-arg zeta
# eta # TODO. related to zeta
]

polygamma2(x)=polygamma(2,x)

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

