reduce1sum = Dict{Symbol,Any}(
:sum => :(dy->dy.+zeros_like(x)),
)
defgrads(reduce1sum, AbstractArray; dymul=false)

reduce2sum = Dict{Symbol,Any}(
:sum => :(dy->dy.+zeros_like(x1))
)
defgrads(reduce2sum, AbstractArray, Integer; dymul=false)

testargs{T1<:AbstractArray,T2<:Number}(::Fn{:sum}, ::Type{T1}, ::Type{T2})=(randn(2,2),1)

# TODO: implement more general sum ops
