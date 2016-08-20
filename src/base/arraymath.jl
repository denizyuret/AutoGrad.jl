arraymath1arg = Dict{Symbol,Any}(
:transpose => :transpose,
:ctranspose => :ctranspose,
)

defgrads(arraymath1arg, AbstractVecOrMat, dymul=false)

# Other functions in arraymath.jl:

# eval
# conj!
# $(Expr(:$, :f)): Not a symbol
# -
# real
# imag
# !
# promote_array_type: Not exported
# ./
# .\
# .^
# +
# slicedim
# flipdim
# rotl90
# rotr90
# rot180
# transpose!
# transposeblock!: Not exported
# ctranspose!
# ctransposeblock!: Not exported
# ccopy!: Not exported
# transpose
# ctranspose
# _cumsum_type: Not exported
# $(Expr(:$, :fp)): Not a symbol
# $(Expr(:$, :f!)): Not a symbol
