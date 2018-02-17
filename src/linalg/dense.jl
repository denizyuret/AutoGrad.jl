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
