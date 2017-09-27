gamma1arg = [
(:gamma, :(y.*digamma_dot(x)), (-Inf,Inf)),
(:lfact, :(sign_dot(y).*digamma_dot(x+1)), (-Inf,Inf)),
(:lgamma, :(digamma_dot(x)), (-Inf,Inf)),
(:digamma, :(trigamma_dot(x)), (-Inf,Inf)), # polygamma(0,x)
(:trigamma, :(polygamma(2,x)), (-Inf,Inf)), # polygamma(1,x)
(:invdigamma, :(1./trigamma_dot(y)), (-Inf,Inf)),
# zeta: TODO. Riemann 1-arg zeta
# eta # TODO. related to zeta
]

for (f,g,r) in gamma1arg
    bf = broadcast_func(f)
    @eval @primitive $f(x),dy,y  (dy.*($g))
    if bf != f
        @eval @primitive $bf(x),dy,y  (dy.*($g))
    end
    addtest1(f,r)
end

gamma2arg = [
#:beta => :TODO,                          # gamma,operators
#:lbeta => :TODO,                         # gamma,operators
#:zeta => :TODO,                 # Hurwitz 2-arg zeta
]

# polygamma wants x1 to be a non-negative integer, x2 unrestricted
@primitive polygamma(x1,x2),dy,y  nothing  unbroadcast(x2,dy.*polygamma(x1+1,x2))
polygamma2(x,i)=polygamma(i,x)
addtest(:polygamma2, randn(), rand(0:5))
# TODO: add broadcasting version
