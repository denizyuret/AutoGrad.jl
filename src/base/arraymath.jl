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
@primitive transpose(x::AbstractVecOrMat)  transpose
# ctranspose
@primitive ctranspose(x::AbstractVecOrMat) ctranspose
# _cumsum_type: Not exported
# cumsum
# cumsum!
# cumsum_pairwise!
# cummin
# cummax
