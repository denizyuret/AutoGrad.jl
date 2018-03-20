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

@primitive triu(x),dy,y  dy.*triu(fill(1.0,size(x)))
addtest(:triu, rand(3,3))

# tril!

@primitive tril(x),dy,y  dy.*tril(fill(1.0,size(x)))
addtest(:tril, rand(3,3))

# gradient
# diagind

# code adapted from https://github.com/FluxML/Flux.jl/pull/169
function _kron(mat1, mat2)
    @assert ndims(mat1) == ndims(mat2) == 2 "Only support matrices for the time being"
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end

kron(a::Rec, b::Rec)  = _kron(a, b)
kron(a::Rec, b) = _kron(a, b)
kron(a, b::Rec) = _kron(a, b)
addtestN(:kron, rand(2,3), rand(4,5))

_diagm(x) = diagm(0 => x)
@primitive _diagm(x),dy,y   diag(dy) 
addtest(:_diagm, rand(3))
@primitive diag(x),dy,y   _diagm(dy)  # alternative: Diagonal(dy)
addtest(:diag, rand(3,3))
# @zerograd Matrix(1.0I, x, x)
# @zerograd eye(x)

if VERSION < v"0.7.0-DEV.3439"
    import Compat.LinearAlgebra.trace
    @primitive trace(x),dy,y  dy*eye(size(x,1),size(x,2))            # alternative: dy*Diagonal(ones(x))
    addtest(:trace, rand(3,3))
else
    @primitive tr(x),dy,y  dy*Matrix(1.0I, size(x,1), size(x,2))     # alternative: dy*Diagonal(ones(x))
    addtest(:tr, rand(3,3))
end

# ^
# expm
# expm!: Not exported
# rcswap!: Not exported
# logm
# sqrtm

# ref https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf 
@primitive inv(x),dy,y  (yt=transpose(y); -yt*dy*yt)
addtest(:inv, rand(2,2))

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

_choltest(x)=chol(x'x)
addtest(:_choltest, rand(3,3))

@primitive lq(x),dy,y   lq_back(y, dy)


# ref: https://arxiv.org/pdf/1710.08717.pdf
function lq_back(y, dy)
    L, Q = y
    dL, dQ = dy
    dL == nothing && (dL = fill(0.0,size(L)))
    dQ == nothing && (dQ = fill(0.0,size(Q)))
    dL = tril(dL)
    M = Symmetric(L'dL - dQ*Q', :L)
    S = inv(L)' *(dQ + M*Q)
    S
end

_lq1(x)=lq(x)[1]; addtest(:_lq1, rand(3,3))
_lq2(x)=lq(x)[2]; addtest(:_lq2, rand(3,3))
_lq3(x)=(y=lq(x); sum(y[1]+y[2])); addtest(:_lq3, rand(3,3))

@primitive qr(x),dy,y   qr_back(y, dy)

function qr_back(y, dy)
    Q, R = y
    dQ, dR = dy
    dR == nothing && (dR = fill(0.0, size(R)))
    dQ == nothing && (dQ = fill(0.0, size(Q)))
    dR = triu(dR)
    M = Symmetric(R*dR' - dQ'*Q, :L)
    S = (dQ + Q*M)*inv(R)'
    S
end

_qr1(x)=qr(x)[1]; addtest(:_qr1, rand(3,3))
_qr2(x)=qr(x)[2]; addtest(:_qr2, rand(3,3))
_qr3(x)=(y=qr(x); sum(y[1]+y[2])); addtest(:_qr3, rand(3,3))
