# vect: Not exported
# oldstyle_vcat_warning: Not exported
# size: interfaces.jl
# eltype: interfaces.jl
# elsize: Not exported
# ndims: interfaces.jl
# length: interfaces.jl
# endof: interfaces.jl
# first: composite using start/next
# last: composite using getindex/endof
# stride: interfaces.jl
# strides: interfaces.jl
# isassigned: interfaces.jl
# trailingsize: Not exported
# linearindexing: Not exported
# checkbounds: interfaces.jl
# throw_boundserror: Not exported
# _internal_checkbounds: Not exported
# similar: interfaces.jl
# reshape
@primitive  reshape(x::AbstractArray,i...)  (dy->reshape(dy,size(x)))
addtest(:reshape, rand(2,2), (1,4))
# copy!: Overwriting operation
# copy: interfaces.jl
# copy_transpose!: Not exported
# zero: interfaces.jl
# start: interfaces.jl
# next: interfaces.jl
# done: interfaces.jl
# eachindex: interfaces.jl
# _maxlength: Not exported
# isempty: interfaces.jl
# convert: Cannot support.
# full
@primitive full(x::AbstractArray) identity
addtest(:full,rand(2,2))
# map: Cannot support.
# pointer: interfaces.jl
# getindex: interfaces.jl
# unsafe_getindex: Not exported
# _getindex: Not exported
# _unsafe_getindex: Not exported
# setindex!: interfaces.jl, not supported
# unsafe_setindex!: Not exported
# _setindex!: Not exported
# _unsafe_setindex!: Not exported
# get (getindex with a default value)
# This can be left as a composite function, it will get its gradient from getindex if necessary.
get{T<:AbstractArray}(A::Node{T}, i::Integer, default) = checkbounds(Bool, length(A), i) ? A[i] : default
get{T<:AbstractArray}(A::Node{T}, I::Tuple{}, default) = similar(A, typeof(default), 0)
get{T<:AbstractArray}(A::Node{T}, I::Dims, default)    = checkbounds(Bool, size(A), I...) ? A[I...] : default
# get!: Overwriting function
# promote_eltype: Not exported

# cat: TODO
# vcat: TODO
# hcat: TODO

# typed_vcat: Not exported
# typed_hcat: Not exported
# cat_t: Not exported

# hvcat: TODO

# hvcat_fill: Not exported
# typed_hvcat: Not exported
# isequal: interfaces.jl
# lexcmp: interfaces.jl
# ==: interfaces.jl
# sub2ind: Not an array op.
# _sub2ind: Not exported
# ind2sub: Not an array op.
# ind2sub!: Not an array op.
# mapslices: Cannot support.
# promote_to!: Not exported
# map_promote: Not exported
# map!: Cannot support.
# map_to!: Not exported
# ith_all: Not exported
# map_n!: Not exported
# map_to_n!: Not exported
# push!: Overwriting function
# unshift!: Overwriting function
# hash: Works for Node.
