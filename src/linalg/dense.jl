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

@primitive triu(x),dy,y  dy.*triu(ones(x))
addtest(:triu, rand(3,3))

# tril!

@primitive tril(x),dy,y  dy.*tril(ones(x))
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

@primitive diagm(x),dy,y   diag(dy) 
addtest(:diagm, rand(3))
@primitive diag(x),dy,y   diagm(dy)  # alternative: Diagonal(dy)
addtest(:diag, rand(3,3))
@zerograd eye(x)
@primitive trace(x),dy,y  dy*eye(x) # alternative: dy*Diagonal(ones(x))
addtest(:trace, rand(3,3))

# ^
# expm
# expm!: Not exported
# rcswap!: Not exported
# logm
# sqrtm

# ref https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf 
@primitive inv(x),dy,y  (yt=y.'; -yt*dy*yt)
addtest(:inv, rand(2,2))

# factorize
# \
# pinv
# nullspace
# cond
# sylvester
# lyap
