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

# ref: formulua in https://arxiv.org/pdf/1602.07527.pdf
# as described in https://arxiv.org/pdf/1710.08717.pdf
# In julia L is upper triangular
function chol_back(L, dL)
    dL = triu(dL)
    iL = inv(L)
    S = iL * Symmetric(L*dL',:L) * iL'
    S/2
end

@primitive lq(x),dy,y   lq_back(y, dy)


# ref: https://arxiv.org/pdf/1710.08717.pdf
function lq_back(y, dy)
    L, Q = y
    dL, dQ = dy
    dL == nothing && (dL = zeros(L))
    dQ == nothing && (dQ = zeros(Q))
    dL = tril(dL)
    M = Symmetric(L'dL - dQ*Q', :L)
    S = inv(L)' *(dQ + M*Q)
    S
end

@primitive qr(x),dy,y   qr_back(y, dy)

function qr_back(y, dy)
    Q, R = y
    dQ, dR = dy
    dR == nothing && (dR = zeros(R))
    dQ == nothing && (dQ = zeros(Q))
    dR = triu(dR)
    M = Symmetric(R*dR' - dQ'*Q, :L)
    S = (dQ + Q*M)*inv(R)'
    S
end



