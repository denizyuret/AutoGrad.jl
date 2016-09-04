number1arg = [
(:abs, :(sign(x))),
(:abs2, :(2x)),
]

for (f,g) in number1arg
    @eval @primitive $f(x),dy,y  (dy.*($g))
    addtest1(f,(-Inf,Inf))
end

number1zero = [
:isinteger,
:sign,
:signbit,
]
for f in number1zero; @eval @zerograd $f(x); end

# TODO:
# size: interfaces.jl
# eltype: interfaces.jl
# ndims: interfaces.jl
# length: interfaces.jl
# endof: interfaces.jl
# getindex: interfaces.jl
# unsafe_getindex: Not exported
# first: compound using start/next
# last: compound using getindex/endof
# divrem
# fldmod
# copysign
# conj
# transpose
# ctranspose
# inv
# angle
# widemul
# start: interfaces.jl
# next: interfaces.jl
# done: interfaces.jl
# isempty: interfaces.jl
# in
# map
# zero: interfaces.jl
# one: interfaces.jl
# factorial
