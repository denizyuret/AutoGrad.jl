reduce1arg = [
(:sum,     :(ones(x))),
(:sumabs_compat,  :(sign_dot(x))),
(:sumabs2_compat, :(2x)),
(:prod,    :(y./x)),
(:maximum, :(y.==x)),
(:minimum, :(y.==x)),
(:maxabs_compat,  :(y.==abs_dot(x))),
(:minabs_compat,  :(y.==abs_dot(x))),
]

if VERSION >= v"0.6-"; @eval begin
    sumabs_compat(x...)=sum(abs,x...)
    sumabs2_compat(x...)=sum(abs2,x...)
    minabs_compat(x...)=minimum(abs,x...)
    maxabs_compat(x...)=maximum(abs,x...)
end; else; @eval begin
    sumabs_compat(x...) = sumabs(x...)
    sumabs2_compat(x...) = sumabs2(x...)
    minabs_compat(x...) = minabs(x...)
    maxabs_compat(x...) = maxabs(x...)
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
