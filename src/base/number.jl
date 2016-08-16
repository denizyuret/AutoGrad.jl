number1arg = Dict{Symbol,Any}(
:abs2 => :(2.*x),               # number,operators
:sign => 0,                     # number,arraymath
)

defgrads(number1arg, Number)
defgrads(number1arg, AbstractArray)
