number1arg = Dict{Symbol,Any}(
:abs2 => :(2.*x),               # number,operators
:sign => 0,                     # number,arraymath
)

defgrads(number1arg, Number)
defgrads(number1arg, AbstractArray)


# TODO:

# eval
# isinteger
# size
# eltype
# ndims
# length
# endof
# getindex
# unsafe_getindex: Not exported
# first
# last
# divrem
# fldmod
# signbit
# sign
# abs
# abs2
# copysign
# conj
# transpose
# ctranspose
# inv
# angle
# widemul
# start
# next
# done
# isempty
# in
# map
# zero
# one
# factorial
