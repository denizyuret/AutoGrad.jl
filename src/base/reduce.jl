reduce1arg = Dict{Symbol,Any}(
:sum => :manualdefinition,
)

@primitive sum{T<:AbstractArray}(x::Node{T})
sum{T<:AbstractArray}(::D1,y::Node,x::Node{T})=(dy->dy.+zeros_like(x))
