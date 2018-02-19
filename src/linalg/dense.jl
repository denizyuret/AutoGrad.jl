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

@primitive diagm(x),dy,y   diag(dy) 
addtest(:diagm, rand(3))
@primitive diag(x),dy,y   diagm(dy)  # alternative: Diagonal(dy)
addtest(:diag, rand(3,3))
@zerograd eye(x)
@primitive trace(x),dy,y  dy*eye(x) # alternative: dy*Diagonal(ones(x))
addtest(:trace, rand(3,3))

# kron
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
