reduce1arg = [
(:sum,      :(_ones(x))),
(:sumabs_,  :(sign.(x))),
(:sumabs2_, :(2x)),
(:mean,     :(_ones(x) .* convert(eltype(x), length(y) / length(x)))),
(:prod,     :(y./x)),
(:maximum,  :(y.==x)),
(:minimum,  :(y.==x)),
(:maxabs_,  :((y.==abs.(x)).*sign.(x))),
(:minabs_,  :((y.==abs.(x)).*sign.(x))),
]
sumabs_(x...; kargs...)=sum(abs,x...; kargs...)
sumabs2_(x...; kargs...)=sum(abs2,x...; kargs...)
minabs_(x...; kargs...)=minimum(abs,x...; kargs...)
maxabs_(x...; kargs...)=maximum(abs,x...; kargs...)

_ones(x::Rec{T}) where T<:Number = one(T) #fix #56
_ones(x::Rec{Array{T}}) where T<:Number = fill(one(T), size(x,value))
_ones(x::Rec) = ones(x)

for (f,g) in reduce1arg
    @eval @primitive  $f(x,i...; kargs...),dy,y   (dy.*($g))
    addtest(f, randn())
    addtest(f, randn(2))
    if VERSION < v"0.7.0-DEV.4064"
        addtest(f, randn(2,2), 1)
        addtest(f, randn(2,2), 2)
        addtest(f, randn(2,2,2,2), (1,2))
    else
        addtest(f, randn(2,2), dims = 1)
        addtest(f, randn(2,2), dims = 2)
        addtest(f, randn(2,2,2,2), dims = (1,2))
    end
    # @eval @primitive $f(x::Tuple),dy,y (x=[x...];tuple((dy.*($g))...))
    # addtest(f, (rand(2)...))
end    

# TODO: more general tuple reduction
@primitive  sum(x::Tuple),dy  ntuple(i->dy,length(x))

# TODO: gradcheck cannot handle tuples yet.
# addtest(:sum, (rand(2)...))

# TODO: implement more general sum ops

let sum_r = recorder(sum), max_r = recorder(maximum), min_r = recorder(minimum)
    global sum, maximum, minimum
    sum(f::typeof(abs), x::Rec, r...; kargs...) = sum_r(f, x, r...; kargs...)
    sum(f::typeof(abs2), x::Rec, r...; kargs...) = sum_r(f, x, r...; kargs...)
    maximum(f::typeof(abs), x::Rec, r...; kargs...) = max_r(f, x, r...; kargs...)
    minimum(f::typeof(abs), x::Rec, r...; kargs...) = min_r(f, x, r...; kargs...)
    sum(::Type{Grad{2}},dy,y,f::typeof(abs),x,r...; kargs...) = sumabs_(Grad{1},dy,y,x,r...; kargs...)
    sum(::Type{Grad{2}},dy,y,f::typeof(abs2),x,r...; kargs...) = sumabs2_(Grad{1},dy,y,x,r...; kargs...)
    maximum(::Type{Grad{2}},dy,y,f::typeof(abs),x,r...; kargs...) = maxabs_(Grad{1},dy,y,x,r...; kargs...)
    minimum(::Type{Grad{2}},dy,y,f::typeof(abs),x,r...; kargs...) = minabs_(Grad{1},dy,y,x,r...; kargs...)
end

# TODO: other functions in reduce.jl:
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
# sum_kbn
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
