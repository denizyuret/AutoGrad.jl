dense2arg = Dict{Symbol,Any}(
#:^ => :todo, # Matrix^Number defined in dense
)

# TODO:

# eval
# scale!
# isposdef!
# isposdef
# stride1: Not exported
# mapreduce_seq_impl: Not exported
# norm
# vecnorm1: Not exported
# vecnorm2: Not exported
# triu!
# triu
# tril!
# tril
# gradient
# diagind
# diag
# diagm
# trace
# kron
# ^
# expm
# expm!: Not exported
# rcswap!: Not exported
# logm
# sqrtm
# inv
# factorize
# \
# pinv
# nullspace
# cond
# sylvester
# lyap

@primitive chol(x),dy,y   chol_back(y, dy)

chol_ϕ(A) = tril(A) - 0.5diagm(diag(A))

# ref: Iain Murray's https://arxiv.org/pdf/1602.07527.pdf
# difference with the paper with respect to the paper:
# julia L is upper triangular and we do not need the
# final simmetrization of S
function chol_back(L, dL)
    dL = triu(dL)
    iL = inv(L)
    S = iL * chol_ϕ(L*dL') * iL'
    #  S + S' - diagm(diag(S))
    S
end


