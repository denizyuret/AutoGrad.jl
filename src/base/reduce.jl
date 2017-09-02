reduce1arg = [
(:sum,     :(ones(x))),
(:sumabs_,  :(sign_dot(x))),
(:sumabs2_, :(2x)),
(:prod,    :(y./x)),
(:maximum, :(y.==x)),
(:minimum, :(y.==x)),
(:maxabs_,  :(y.==abs_dot(x))),
(:minabs_,  :(y.==abs_dot(x))),
]

if VERSION >= v"0.6-"; @eval begin
    sumabs_(x...)=sum(abs,x...)
    sumabs2_(x...)=sum(abs2,x...)
    minabs_(x...)=minimum(abs,x...)
    maxabs_(x...)=maximum(abs,x...)
end; else; @eval begin
    sumabs_(x...) = sumabs(x...)
    sumabs2_(x...) = sumabs2(x...)
    minabs_(x...) = minabs(x...)
    maxabs_(x...) = maxabs(x...)
end; end

for (f,g) in reduce1arg
    @eval @primitive $f(x,i...),dy,y   (dy.*($g))
    addtest(f, rand(2))
    addtest(f, rand(2,2), 1)
    addtest(f, rand(2,2), 2)
    # @eval @primitive $f(x::Tuple),dy,y (x=[x...];tuple((dy.*($g))...))
    # addtest(f, (rand(2)...))
end    

# TODO: more general tuple reduction
@primitive  sum(x::Tuple),dy  ntuple(i->dy,length(x))

# TODO: gradcheck cannot handle tuples yet.
# addtest(:sum, (rand(2)...))

# TODO: implement more general sum ops

if VERSION >= v"0.6-"
    let sum_r = recorder(sum), max_r = recorder(maximum), min_r = recorder(minimum)
        global sum, maximum, minimum
        sum(f::typeof(abs), x::Rec, r...) = sum_r(f, x, r...)
        sum(f::typeof(abs2), x::Rec, r...) = sum_r(f, x, r...)
        maximum(f::typeof(abs), x::Rec, r...) = max_r(f, x, r...)
        minimum(f::typeof(abs), x::Rec, r...) = min_r(f, x, r...)
        sum(Grad{2},dy,y,f::typeof(abs),x,r...) = sumabs(Grad{1},dy,y,x,r...)
        sum(Grad{2},dy,y,f::typeof(abs2),x,r...) = sumabs2(Grad{1},dy,y,x,r...)
        maximum(Grad{2},dy,y,f::typeof(abs),x,r...) = maxabs(Grad{1},dy,y,x,r...)
        minimum(Grad{2},dy,y,f::typeof(abs),x,r...) = minabs(Grad{1},dy,y,x,r...)
    end
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
