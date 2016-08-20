# This is different from AbstractArray:
next{T<:Array}(a::Node{T},i) = (a[i],i+1)

# Other functions in array.jl:
#
# eval
# call
# cconvert: Not exported
# size
# asize_from: Not exported
# length
# elsize: Not exported
# sizeof
# strides
# isassigned
# unsafe_copy!
# copy!
# copy
# reinterpret
# reshape
# similar
# getindex
# fill!
# fill
# cell
# $(Expr(:$, :fname)): Not a symbol
# eye
# one
# convert
# promote_rule
# collect
# start
# next
# done
# unsafe_getindex: Not exported
# setindex!
# unsafe_setindex!: Not exported
# _growat!: Not exported
# _growat_beg!: Not exported
# _growat_end!: Not exported
# _deleteat!: Not exported
# _deleteat_beg!: Not exported
# _deleteat_end!: Not exported
# push!
# append!
# prepend!
# resize!
# sizehint!
# pop!
# unshift!
# shift!
# insert!
# deleteat!
# splice!
# empty!
# lexcmp
# reverse
# reverseind
# reverse!
# vcat
# hcat
# findnext
# findfirst
# findprev
# findlast
# find
# findn
# findnz
# findmax
# findmin
# indmax
# indmin
# indexin
# findin
# indcopy: Not exported
# filter
# filter!
# intersect
# union
# setdiff
# symdiff
