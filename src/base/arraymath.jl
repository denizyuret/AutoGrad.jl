# conj!: Overwriting operation
# -
# ~
# conj
# sign
# real
# imag
# !
# promote_array_type: Not exported
# ./
# .\
# .^
# +
# div
# mod
# &
# |
# $
# .+
# .-
# .*
# .%
# .<<
# .>>
# rem
# slicedim
# flipdim
# rotl90
# rotr90
# rot180
# transpose!: Overwriting
# transposeblock!: Not exported
# ctranspose!: Overwriting
# ctransposeblock!: Not exported
# ccopy!: Not exported
# transpose
@primitive transpose(x),dy  transpose(dy)
addtest(transpose,rand(2,2))
# ctranspose
@primitive ctranspose(x),dy ctranspose(dy)
addtest(ctranspose,rand(2,2))
# _cumsum_type: Not exported
# cumsum
# cumsum!
# cumsum_pairwise!
# cummin
# cummax
