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
addtest(:kron, rand(2,3), rand(4,5))

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
