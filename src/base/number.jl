number1arg = [
    (:abs, :(sign.(x))),
    (:abs2, :(2x)),
]

for (f,g) in number1arg
    let bf = broadcast_func(f)
        @eval @primitive $f(x),dy,y  (dy.*($g))
        if bf != f
            @eval @primitive $bf(x),dy,y  (dy.*($g))
        end
        addtest1(f,(-Inf,Inf))
    end
end

number1zero = [
:isinteger,
:sign,
:signbit,
]
for f in number1zero
    let bf = broadcast_func(f)
        @eval @zerograd $f(x)
        if bf != f
            @eval @zerograd $bf(x)
        end
    end
end

# TODO:
# size: interfaces.jl
# eltype: interfaces.jl
# ndims: interfaces.jl
# length: interfaces.jl
# lastindex: interfaces.jl
# getindex: interfaces.jl
# unsafe_getindex: Not exported
# first: compound using start/next
# last: compound using getindex/lastindex
# divrem
# fldmod
# copysign
# conj
# transpose
# adjoint
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
