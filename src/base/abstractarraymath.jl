# isinteger
# isreal
# ctranspose
# transpose
# vec
@primitive vec(x),dy reshape(dy,size(x))
addtest(vec,rand(2,2))
# _sub: Not exported
# squeeze
# conj
# conj!
# real
# imag
# +
# *
# /
# \
# slicedim
# flipdim
# circshift
# cumsum_kbn
# ipermutedims
@primitive permutedims(x,dims),dy ipermutedims(dy,dims)
@primitive ipermutedims(x,dims),dy permutedims(dy,dims)
addtest(permutedims,rand(2,3,4),(3,1,2))
addtest(ipermutedims,rand(2,3,4),(3,1,2))
# repmat
# repeat

