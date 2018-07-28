# Functions defined in julia-0.7.0-beta2/base/reduce.jl:
import Base: all, any, count, foldl, foldr, mapfoldl, mapfoldr, mapreduce, maximum, minimum, prod, reduce, sum
# Not exported: _all, _any, _empty_reduce_error, _mapreduce, add_sum, mapfoldl_impl, mapfoldr_impl, mapreduce_empty, mapreduce_empty_iter, mapreduce_first, mapreduce_impl, mul_prod, pairwise_blocksize, reduce_empty, reduce_first

reduce1arg = [
(:sum,      :(_ones(x))),
(:sumabs_,  :(sign.(x))),
(:sumabs2_, :(2x)),
#TODO (:mean,     :(_ones(x) .* convert(eltype(x), length(y) / length(x)))),
(:prod,     :(y./x)),
(:maximum,  :(y.==x)),
(:minimum,  :(y.==x)),
(:maxabs_,  :((y.==abs.(x)).*sign.(x))),
(:minabs_,  :((y.==abs.(x)).*sign.(x))),
]
sumabs_(x...)=sum(abs,x...)
sumabs2_(x...)=sum(abs2,x...)
minabs_(x...)=minimum(abs,x...)
maxabs_(x...)=maximum(abs,x...)

_ones(x::Rec{T}) where T<:Number = one(T) #fix #56
_ones(x::Rec) = fill!(similar(x.value),1) # ones(x)

for (f,g) in reduce1arg
    @eval @primitive  $f(x,i...),dy,y   (dy.*($g))
    addtest(f, randn())
    addtest(f, randn(2))
    addtest(f, randn(2,2), 1)
    addtest(f, randn(2,2), 2)
    addtest(f, randn(2,2,2,2), (1,2))
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
    sum(f::typeof(abs), x::Rec, r...) = sum_r(f, x, r...)
    sum(f::typeof(abs2), x::Rec, r...) = sum_r(f, x, r...)
    maximum(f::typeof(abs), x::Rec, r...) = max_r(f, x, r...)
    minimum(f::typeof(abs), x::Rec, r...) = min_r(f, x, r...)
    sum(::Type{Grad{2}},dy,y,f::typeof(abs),x,r...) = sumabs_(Grad{1},dy,y,x,r...)
    sum(::Type{Grad{2}},dy,y,f::typeof(abs2),x,r...) = sumabs2_(Grad{1},dy,y,x,r...)
    maximum(::Type{Grad{2}},dy,y,f::typeof(abs),x,r...) = maxabs_(Grad{1},dy,y,x,r...)
    minimum(::Type{Grad{2}},dy,y,f::typeof(abs),x,r...) = minabs_(Grad{1},dy,y,x,r...)
end


