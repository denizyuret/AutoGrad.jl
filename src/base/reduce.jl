reduce1arg = Dict{Symbol,Any}(
:sum => :(dy->dy.+zeros_like(x)),
)

defgrads(reduce1arg, AbstractArray; dymul=false)

reduce2arg = Dict{Symbol,Any}(
:sum => (:(dy->dy.+zeros_like(x1)),0),
)
defgrads(reduce2arg, AbstractArray, Integer; dymul=false)

# TODO: implement more general sum ops
