import LinearAlgebra: *, adjoint, det, diag, diagm, dot, inv, kron, logabsdet, logdet, lq, norm, qr, svd, tr, transpose, tril, triu

# julia/stdlib/v0.7/LinearAlgebra/src/LinearAlgebra.jl Functions:
# axpy!
# axpby!
# bunchkaufman
# bunchkaufman!
# @primitive chol(x),dy,y   chol_back(y, dy) # `chol(A::AbstractMatrix)` is deprecated, use `(cholesky(A)).U` instead.
# cholesky
# cholesky!
# cond
# condskeel
# copyto!
# copy_transpose!
# cross
@primitive adjoint(x),dy    adjoint(dy)
# adjoint!
@primitive det(x),dy,y  dy*y*inv(x)'
@primitive diag(x),dy,y   diagm(0=>dy)  # alternative: Diagonal(dy)
@primitive diag(x,i),dy,y   diagm(i=>dy)  # warning: these only works for square matrices
# diagind
# @primitive diagm(x),dy,y   diag(dy,x[1]) # TODO: diagm has a pair input
@primitive dot(x1, x2),dy,y  dy*x2  dy*x1  # addtestN(:dot, rand(3,2), rand(3,2))
# eigen
# eigen!
# eigmax
# eigmin
# eigvals
# eigvals!
# eigvecs
# @zerograd eye(x) # Warning: `eye(A::AbstractMatrix{T}) where T` has been deprecated in favor of `I` and `Matrix` constructors. For a direct replacement, consider `Matrix{eltype(A)}(I, size(A))`.If `eltype(A)` element type is not necessary, consider the shorter `Matrix(I, size(A))` (with default `eltype(I)` `Bool`).
# factorize
# givens
# hessenberg
# hessenberg!
# inv: not part of LinearAlgebra, imported from Base
@primitive inv(x),dy,y  (yt=y'; -yt*dy*yt) # addtest(:inv, rand(2,2)) # ref https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf 
# isdiag
# ishermitian
# isposdef
# isposdef!
# issuccess
# issymmetric
# istril
# istriu
kron(a::Value, b::Value)  = _kron(a, b)
kron(a::Value, b) = _kron(a, b)
kron(a, b::Value) = _kron(a, b)
# ldiv!
# ldlt!
# ldlt
@primitive logabsdet(x),dy,y  dy[1]*inv(x)'
@primitive logdet(x),dy,y  dy*inv(x)'
# lowrankdowndate
# lowrankdowndate!
# lowrankupdate
# lowrankupdate!
# lu
# lu!
# lyap
# mul!
# lmul!
# rmul!
@primitive norm(x),dy,y  normback(x,2,dy,y)
@primitive norm(x,p::Real),dy,y normback(x,p,dy,y)
# normalize
# normalize!
# nullspace
# ordschur!
# ordschur
# pinv
@primitive qr(x),dy,y   qr_back(y, dy)
# qr!
@primitive lq(x),dy,y   lq_back(y, dy)
# lq!
# opnorm
# rank
# rdiv!
# schur
# schur!
@primitive svd(x),dy,y  svd_back(x, y, dy)
# svd!
# svdvals!
# svdvals
# sylvester
@primitive tr(x),dy,y  dy*Matrix(I,size(x)) # tr: trace deprecated using tr instead # addtest(:tr, rand(3,3))
@primitive transpose(x),dy  transpose(dy)
# transpose!
# transpose_type
@primitive tril(x),dy,y  dy.*tril(fill!(similar(x),1))  # addtest(:tril, rand(3,3))
# tril!
@primitive triu(x),dy,y  dy.*triu(fill!(similar(x),1))  # addtest(:triu, rand(3,3))
# triu!

# julia/stdlib/v0.7/LinearAlgebra/src/LinearAlgebra.jl Operators:
# @primitive1 *(x1,x2),dy  (dy*x2')  (x1'*dy) # --> base.jl
# @primitive1 ^(x1,x2::Integer),dy,y  error("Derivatives of integer matrix powers not defined.") # this gives real results, TODO. --> base.jl
# @primitive1 ^(x1,x2::Number),dy,y   error("Derivatives of real matrix powers not defined.") # this could give imaginary results, out of scope --> base.jl
# \
# /
# ⋅ = dot
# × = cross
# +
# -
# ==

# julia/stdlib/v0.7/LinearAlgebra/src/blas.jl exports:
# # Level 1
# asum
# axpy!
# axpby!
# blascopy!
# dot
# dotc
# dotu
# scal!
# scal
# nrm2
# iamax
# # Level 2
# gbmv!
# gbmv
# gemv!
# gemv
# hemv!
# hemv
# sbmv!
# sbmv
# symv!
# symv
# trsv!
# trsv
# trmv!
# trmv
# ger!
# syr!
# her!
# # Level 3
# herk!
# herk
# her2k!
# her2k
# gemm!
# gemm
# symm!
# symm
# hemm!
# hemm
# syrk!
# syrk
# syr2k!
# syr2k
# trmm!
# trmm
# trsm!
# trsm


### Helper functions

# chol: Warning: `chol(A::AbstractMatrix)` is deprecated, use `(cholesky(A)).U` instead.
# ref: formulua in https://arxiv.org/pdf/1602.07527.pdf
# as described in https://arxiv.org/pdf/1710.08717.pdf
# In julia L is upper triangular
function chol_back(L, dL)
    dL = triu(dL)
    iL = inv(L)
    S = iL * Symmetric(L*dL',:L) * iL'
    S/2
end
# _choltest(x)=chol(x'x)
# addtest(:_choltest, rand(3,3))

# kron
# code adapted from https://github.com/FluxML/Flux.jl/pull/169
function _kron(mat1, mat2)
    @assert ndims(mat1) == ndims(mat2) == 2 "Only support matrices for the time being"
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end
# addtestN(:kron, rand(2,3), rand(4,5))

# qr
function qr_back(y, dy)
    Q, R = y
    dQ, dR = dy
    dR == nothing && (dR = zero(R))
    dQ == nothing && (dQ = zero(Q))
    dR = triu(dR)
    M = Symmetric(R*dR' - dQ'*Q, :L)
    S = (dQ + Q*M)*inv(R)'
    S
end
# _qr1(x)=qr(x)[1]; addtest(:_qr1, rand(3,3))
# _qr2(x)=qr(x)[2]; addtest(:_qr2, rand(3,3))
# _qr3(x)=(y=qr(x); sum(y[1]+y[2])); addtest(:_qr3, rand(3,3))

# lq
# ref: https://arxiv.org/pdf/1710.08717.pdf
function lq_back(y, dy)
    L, Q = y
    dL, dQ = dy
    dL == nothing && (dL = zero(L))
    dQ == nothing && (dQ = zero(Q))
    dL = tril(dL)
    M = Symmetric(L'dL - dQ*Q', :L)
    S = inv(L)' *(dQ + M*Q)
    S
end
# _lq1(x)=lq(x)[1]; addtest(:_lq1, rand(3,3))
# _lq2(x)=lq(x)[2]; addtest(:_lq2, rand(3,3))
# _lq3(x)=(y=lq(x); sum(y[1]+y[2])); addtest(:_lq3, rand(3,3))


# vecnorm has been renamed norm:

function normback(x,p,dy,y)
    if length(x) == 0
        similar(x)
    elseif p == 2
        (dy/y)*x
    elseif p == 1
        dy * sign.(x)
    elseif p == Inf
        dy * sign.(x) .* (abs.(x) .== y)
    elseif p == 0
        zero(x)
    elseif p == -Inf
        dy * sign.(x) .* (abs.(x) .== y)
    else
        (dy*y^(1-p)*(abs.(x).^(p-1)).*sign.(x))
    end
end

# addtest(:norm, rand(2,2)-0.5)
# for p in (0,1,2,Inf,-Inf,rand(),-rand(),1+rand(),-1-rand())
#     addtest(:norm, rand(2,2)-0.5, p)
# end


# det, logdet, logabsdet
#ref https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf 
# TODO make more efficient using the intermediate
# results of SVD in the forward pass

# addtest(:det, rand(3,3))
# addtest(:logdet, 5eye(3) + rand(3,3))
# gradcheck(logabsdet, rand([-1,1]) .* rand(3,3))

# ref https://j-towns.github.io/papers/svd-derivative.pdf
function svd_back(x, y, dy)
    U, s, V = y
    dU, ds, dV = dy

    F = s'.^2 .- s.^2 
    F = 1 ./ (F + eye(F)) - eye(F) #avoid infinities on the diagonal

    dx = zero(x)
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

# _svd1(x)=svd(x)[1]
# _svd2(x)=svd(x)[2]
# _svd3(x)=svd(x)[3]
# for A in (rand(2,2), rand(2,3), rand(3,2))
#     addtest(:_svd1, A)
#     addtest(:_svd2, A)
#     addtest(:_svd3, A)
# end
