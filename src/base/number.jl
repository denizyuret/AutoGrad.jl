number1arg = Dict{Symbol,Any}(
:abs2 => :(2.*x),               # number,operators
:sign => 0,                     # number,arraymath
#:angle => :todo, # angle(z::Real) = atan2(zero(z), z); angle(z::Complex) = atan2(imag(z), real(z)); number,operators
#:conj => :todo,  # number,abstractarraymath
)

defgrads(number1arg, Number)
defgrads(number1arg, AbstractArray)
