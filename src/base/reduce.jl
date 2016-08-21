@primitive  sum(x::AbstractArray,i...)  (dy->dy.+zeros(x))
fixdomain(::Fn{:sum},x...)=(rand()<0.5 ? (rand(2,2),) : (rand(2,2),1))

# reduce1sum = Dict{Symbol,Any}(
# :sum => :(dy->dy.+zeros(x)),
# )
# defgrads(reduce1sum, AbstractArray; dymul=false)

# reduce2sum = Dict{Symbol,Any}(
# :sum => (:(dy->dy.+zeros(x1)),0)
# )

# sum{T<:Integer}(x1::BitArray,x2::Node{T})=sum(x1,x2.value) # To avoid clash with bitarray.jl:1501.
# defgrads(reduce2sum, AbstractArray, Integer; dymul=false)

# testargs{T1<:AbstractArray,T2<:Number}(::Fn{:sum}, ::Type{T1}, ::Type{T2})=(randn(2,2),1)

# Base.zeros(x::Node)=zeros(x.value)

# TODO: implement more general sum ops

# TODO:

# eval
# r_promote: Not exported
# mapfoldl_impl: Not exported
# mapfoldl
# foldl
# mapfoldr_impl: Not exported
# mapfoldr
# foldr
# mapreduce_seq_impl: Not exported
# mapreduce_pairwise_impl: Not exported
# mapreduce
# mapreduce_impl: Not exported
# mr_empty: Not exported
# _mapreduce: Not exported
# reduce
# shortcircuits: Not exported
# shorted: Not exported
# sc_finish: Not exported
# mapreduce_sc_impl: Not exported
# mapreduce_no_sc: Not exported
# mapreduce_sc: Not exported
# sum_pairwise_blocksize: Not exported
# sum
# sumabs
# sumabs2
# sum_kbn
# prod
# maximum
# minimum
# maxabs
# minabs
# extrema
# any
# all
# in
# ∉
# ∋
# ∌
# contains
# count
# call
# countnz
