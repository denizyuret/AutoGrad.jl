# Most functions defined in abstractarray.jl have zerograds.
# So we just unbox the first input.

abstractarray1arg = [
:eltype,                                     
:ndims,                                     
:length,
:endof,
:strides,
:zero,
:isempty,
]

for k in abstractarray1arg
    @eval $k{T<:AbstractArray}(a::Node{T})=$k(a.value)
end

abstractarrayNarg = [
:size,
:stride,
:isassigned,
:checkbounds,
:similar,
:eachindex,
:pointer,
]

for k in abstractarrayNarg
    @eval $k{T<:AbstractArray}(a::Node{T}, i...)=$k(a.value,i...)
end

# These have nonzero grads but can be defined in terms of getindex
first{T<:AbstractArray}(a::Node{T})=getindex(a,first(eachindex(a.value)))
last{T<:AbstractArray}(a::Node{T})=getindex(a,endof(a.value))

@primitive reshape{T<:AbstractArray}(x::Node{T},i...)
reshape{T<:AbstractArray}(::D1,y::Node{T},x::Node{T},i...)=(dy->reshape(dy,size(x)))

# copy is unnecessary in a functional autograd, may be useful with overwriting.
@primitive copy{T<:AbstractArray}(x::Node{T})
copy{T<:AbstractArray}(::D1,y::Node{T},x::Node{T})=identity

# full is identity for AbstractArray, does it really need to be recorded?
@primitive full{T<:AbstractArray}(x::Node{T})
full{T<:AbstractArray}(::D1,y::Node{T},x::Node{T})=identity

# start and done defined for Node{T} in interfaces.jl, we just need to define next.
# start{T<:AbstractArray}(a::Node{T})=start(a.value)   # Returns the initial iteration state
# done{T<:AbstractArray}(a::Node{T},i)=done(a.value,i) # Tests if there are any items remaining

# next: Returns the current item and the next state
# It calls getindex which handles recording and gradients.
# Iteration for AbstractArray uses state=(eachindex(A),curr_idx)
# where eachindex is an iterable of indices
next{T<:AbstractArray}(a::Node{T},i)=((idx, s) = next(i[1], i[2]); (a[idx], (i[1], s)))

# getindex defined for Node{T} in interfaces.jl, we just need to
# define ungetindex. If y=getindex(x,i...) and we receive dy, we need
# to create dx as with zeros similar to x, with only dx[i] set to dy.
# For cell arrays we represent zero entries with `nothing`.
ungetindex{T}(x::AbstractArray{T}, dy, i...)=
    (dx = isbits(T) ? zeros(x) : fill!(Array(Any, size(x)), nothing); setindex!(dx,dy,i...); dx)

## get (getindex with a default value) ##
# This can be left as a composite function, it will get its gradient from getindex if necessary.
get{T<:AbstractArray}(A::Node{T}, i::Integer, default) = checkbounds(Bool, length(A), i) ? A[i] : default
get{T<:AbstractArray}(A::Node{T}, I::Tuple{}, default) = similar(A, typeof(default), 0)
get{T<:AbstractArray}(A::Node{T}, I::Dims, default) = checkbounds(Bool, size(A), I...) ? A[I...] : default

# TODO: 
# cat
# vcat
# hcat
# hvcat

abstractarray2cmp = Dict{Symbol,Any}(
:isequal => 0,
:lexcmp => 0,
:(==) => 0,
)

defgrads(abstractarray2cmp, AbstractArray, AbstractArray)

### Other functions in abstractarray.jl:
#
# vect: Not exported
# oldstyle_vcat_warning: Not exported
# elsize: Not exported
# trailingsize: Not exported
# linearindexing: Not exported
# throw_boundserror: Not exported
# _internal_checkbounds: Not exported
# copy!: Overwriting operation
# copy_transpose!: Not exported
# _maxlength: Not exported
# convert: This may be dangerous
# map: Mapping
# unsafe_getindex: Not exported
# _getindex: Not exported
# _unsafe_getindex: Not exported
# setindex!: Overwriting function
# unsafe_setindex!: Not exported
# _setindex!: Not exported
# _unsafe_setindex!: Not exported
# get!: Overwriting
# promote_eltype: Not exported
# typed_vcat: Not exported
# typed_hcat: Not exported
# cat_t: Not exported
# hvcat_fill: Not exported
# typed_hvcat: Not exported
# sub2ind: Not an array function
# _sub2ind: Not exported
# ind2sub: Not an array function
# ind2sub!: Not an array function
# mapslices: Mapping
# promote_to!: Not exported
# map_promote: Not exported
# map!: Mapping
# map_to!: Not exported
# ith_all: Not exported
# map_n!: Not exported
# map_to_n!: Not exported
# push!: Overwriting
# unshift!: Overwriting
# hash: No need
