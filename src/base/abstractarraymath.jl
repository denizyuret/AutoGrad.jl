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
let perm_r=recorder(permutedims), iperm_r=recorder(ipermutedims)
    global permutedims, ipermutedims
    permutedims(x::Rec,dims) = perm_r(x,dims)
    ipermutedims(x::Rec,dims) = iperm_r(x,dims)
    permutedims(::Type{Grad{1}},dy,y,x::Rec,dims)=ipermutedims(dy,dims)
    ipermutedims(::Type{Grad{1}},dy,y,x::Rec,dims)=permutedims(dy,dims)
end
addtest(permutedims,rand(2,3,4),(3,1,2))
addtest(ipermutedims,rand(2,3,4),(3,1,2))
# repmat
# repeat

