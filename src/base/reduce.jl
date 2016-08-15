reduce1sum = Dict{Symbol,Any}(
:sum => :(dy->dy.+zeros_like(x)),
)

defgrads(reduce1sum, AbstractArray; dymul=false)

reduce2sum = Dict{Symbol,Any}(
:sum => :(dy->dy.+zeros_like(x1)),
)
defgrads(reduce2sum, AbstractArray, Integer; dymul=false)

# TODO: implement more general sum ops
