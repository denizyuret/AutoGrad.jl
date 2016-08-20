next{T<:Tuple}(a::Node{T},i)=(a[i],i+1)
ungetindex(x::Tuple, dy, i) = ntuple(j->(j==i ? dy : nothing), length(x))

tuple1arg = [
:length,
:endof,             
:eltype,                                     
:isempty,
]

for k in tuple1arg
    @eval $k{T<:Tuple}(a::Node{T})=$k(a.value)
end

# TODO:

# eval
# length
# endof
# size
# getindex
# start
# done
# next
# indexed_next: Not exported
# eltype
# ntuple
# map
# heads: Not exported
# tails: Not exported
# isequal
# ==
# hash
# isless
# isempty
# revargs: Not exported
# reverse
# sum
# prod
# all
# any
