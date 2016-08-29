@zerograd sign(x::AorN)
@primitive abs2(x::AorN) (dy->dy.*2.*x)

# number1arg = Dict{Symbol,Any}(
# :abs2 => :(2.*x),               # number,operators
# :sign => 0,                     # number,arraymath
# )

# defgrads(number1arg, Number)
# defgrads(number1arg, AbstractArray)

# start(x::Number) = false
# next(x::Number, state) = (x, true)
# done(x::Number, state) = state
# next{T<:Number}(x::Value{T}, state) = (x, true)

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
