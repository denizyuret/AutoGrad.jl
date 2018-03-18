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
        dy * sign.(x)
    elseif p == Inf
        dy * sign.(x) .* (abs.(x) .== y)
    elseif p == 0
        zeros(x)
    elseif p == -Inf
        dy * sign.(x) .* (abs.(x) .== y)
    else
        (dy*y^(1-p)*(abs.(x).^(p-1)).*sign.(x))
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

#ref https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf 
# TODO make more efficient using the intermediate
# results of SVD in the forward pass
@primitive det(x),dy,y  dy*y*inv(x).'
addtest(:det, rand(3,3))
@primitive logdet(x),dy,y  dy*inv(x).'
addtest(:logdet, 5eye(3) + rand(3,3))
@primitive logabsdet(x),dy,y  dy[1]*inv(x).'
gradcheck(logabsdet, rand([-1,1]) .* rand(3,3))

# isapprox


@primitive svd(x),dy,y  svd_back(x, y, dy)

# ref https://j-towns.github.io/papers/svd-derivative.pdf
function svd_back(x, y, dy)
    U, s, V = y
    dU, ds, dV = dy

    F = s'.^2 .- s.^2 
    F = 1 ./ (F + eye(F)) - eye(F) #avoid infinities on the diagonal

    dx = zeros(x)
    S = diagm(s)
    if ds != nothing
        dx += U*diagm(ds)*V' 
    end
    if dU != nothing
        UUt = U*U'
        dx += (U*(F.*(U'dU-dU'U))*S + (eye(UUt) - UUt)*dU*inv(S))*V'
    end

    if dV != nothing
        VVt = V*V'
        dx += U*(S*(F.*(V'dV-dV'V))*V' + inv(S)*dV'*(eye(VVt) - VVt))
    end

    dx
end

_svd1(x)=svd(x)[1]
_svd2(x)=svd(x)[2]
_svd3(x)=svd(x)[3]
for A in (rand(2,2), rand(2,3), rand(3,2))
    addtest(:_svd1, A)
    addtest(:_svd2, A)
    addtest(:_svd3, A)
end
