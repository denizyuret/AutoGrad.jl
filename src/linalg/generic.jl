generic2arg = Dict{Symbol,Any}(
# :/ => (:(1./x2),:(-x1./abs2(x2))), # (*,N) (V,V) (M,M)
# :\ => (:(-x2./abs2(x1)),:(1./x1)), # (N,*) (V,V) (M,M) (M,V) (V,M)
)

function vecnormback(x,p,dy,y)
    if length(x) == 0
        similar(x)
    elseif p == 2
        (dy/y)*x
    elseif p == 1
        dy * sign(x)
    elseif p == Inf
        dy * sign(x) .* (abs(x) .== y)
    elseif p == 0
        zeros(x)
    elseif p == -Inf
        dy * sign(x) .* (abs(x) .== y)
    else
        (dy*y^(1-p)*(abs(x).^(p-1)).*sign(x))
    end
end

@primitive vecnorm(x),dy,y  vecnormback(x,2,dy,y)
addtest(:vecnorm, rand(2,2)-0.5)
@primitive vecnorm(x,p::Real),dy,y vecnormback(x,p,dy,y)
for p in (0,1,2,Inf,-Inf,rand(),-rand(),1+rand(),-1-rand())
    addtest(:vecnorm, rand(2,2)-0.5, p)
end

# TODO:

# eval
# scale
# generic_scale!: Not exported
# scale!
# cross
# triu
# tril
# triu!
# tril!
# diff
# gradient
# diag
# generic_vecnormMinusInf: Not exported
# generic_vecnormInf: Not exported
# generic_vecnorm1: Not exported
# norm_sqr: Not exported
# generic_vecnorm2: Not exported
# generic_vecnormp: Not exported
# vecnormMinusInf: Not exported
# vecnormInf: Not exported
# vecnorm1: Not exported
# vecnorm2: Not exported
# vecnormp: Not exported
# vecnorm
# norm
# norm1: Not exported
# norm2: Not exported
# normInf: Not exported
# vecdot
# dot
# rank
# trace
# inv
# \
# /
# cond
# condskeel
# issym
# ishermitian
# istriu
# istril
# isdiag
# linreg
# peakflops
# axpy!: Not exported
# reflector!: Not exported
# reflectorApply!: Not exported
# det
# logdet
# logabsdet
# isapprox
