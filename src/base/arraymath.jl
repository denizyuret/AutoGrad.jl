arraymath1arg = Dict{Symbol,Any}(
:transpose => :transpose,
:ctranspose => :ctranspose,
)

defgrads(arraymath1arg, StridedMatrix, dymul=false)
