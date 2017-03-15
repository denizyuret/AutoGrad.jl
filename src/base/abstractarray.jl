# vect: Not exported
# oldstyle_vcat_warning: Not exported
# size: interfaces.jl
# eltype: interfaces.jl
# elsize: Not exported
# ndims: interfaces.jl
# length: interfaces.jl
# endof: interfaces.jl
# first: compound using start/next
# last: compound using getindex/endof
# stride: interfaces.jl
# strides: interfaces.jl
# isassigned: interfaces.jl
# trailingsize: Not exported
# linearindexing: Not exported but part of AbstractArray interface
import Base: linearindexing
@zerograd linearindexing(x)
# checkbounds: interfaces.jl
# throw_boundserror: Not exported
# _internal_checkbounds: Not exported
# similar: interfaces.jl
# reshape
@primitive reshape(x,i...),dy  reshape(dy,size(x))
addtest(reshape,rand(2,2),(4,1))
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
@primitive full(x),dy dy
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
get{T<:AbstractArray}(A::Rec{T}, i::Integer, default) = (if checkbounds(Bool, length(A), i); A[i]; else; default; end)
get{T<:AbstractArray}(A::Rec{T}, I::Tuple{}, default) = similar(A, typeof(default), 0)
get{T<:AbstractArray}(A::Rec{T}, I::Dims, default)    = (if checkbounds(Bool, size(A), I...); A[I...]; else; default; end)
# get!: Overwriting function
# promote_eltype: Not exported

# cat(dims,A...): Concatenate the input arrays along the specified
# dimensions in the iterable dims. For each dimension not in dims, all
# input arrays should have the same size, which will also be the size
# of the output array along that dimension. For each dimension in
# dims, the size of the output array is the sum of the sizes of the
# input arrays along that dimension. If dims is a single number, the
# different arrays are tightly stacked along that dimension. If dims
# is an iterable containing several dimensions, this allows one to
# construct block diagonal matrices and their higher-dimensional
# analogues by simultaneously increasing several dimensions for every
# new input array and putting zero blocks elsewhere. For example,
# cat([1,2], matrices...) builds a block diagonal matrix, i.e. a block
# matrix with matrices[1], matrices[2], ... as diagonal blocks and
# matching zero blocks away from the diagonal.

# After dims, cat can take 0 or more arguments of any type.  We only
# catch the cases where one of the first two args is a Rec.  We
# leave the type of the first arg unspecified, which can be an Int,
# Tuple{Int}, or Vector{Int} and is never boxed.  We assume
# cat(Grad{1},...) will never be called.

typealias CatDims Union{Int,Tuple{Int},Vector{Int}} # julia4 gives ambiguity warnings if first arg type not specified
let cat_r = recorder(cat)
    global cat
    cat(d::CatDims,a::Rec,b::Rec,c...)=cat_r(d,a,b,c...)
    cat(d::CatDims,a,b::Rec,c...)     =cat_r(d,a,b,c...)
    cat(d::CatDims,a::Rec,b...)       =cat_r(d,a,b...)
end
cat(::Type{Grad{1}},y1,y,dims,x...)=nothing
cat{N}(::Type{Grad{N}},y1,y,dims,x...)=uncat(y1,N-1,dims,x...)   # N-1 because first arg is catdims

# For the gradient, we need to extract the n'th block from dy which
# has the same shape as y=cat(dims,x...).  Note that the inputs x[i]
# may have fewer dimensions than dy, or even be scalars.  In those
# cases Julia assumes the missing dimensions are 1.  We need to
# reshape dx to the same size as x.

function uncat(y1,n,dims,x...)
    idx = []
    @inbounds for d=1:ndims(y1)
        if in(d,dims)
            pos = 0
            @inbounds for j=1:n-1
                pos += size(x[j],d)
            end
            push!(idx,(pos+1):(pos+size(x[n],d)))
        else
            push!(idx,1:size(y1,d))
        end
    end
    x1 = y1[idx...]
    if isa(getval(x[n]),Number)
        length(x1)==1 || error("uncat mismatch")
        x1 = x1[1]
    else
        x1 = reshape(x1, size(x[n]))
    end
    return x1
end

function uncat1(x2,y1,n,dims,x...)
    idx = []
    @inbounds for d=1:ndims(y1)
        if in(d,dims)
            pos = 0
            @inbounds for j=1:n-1
                pos += size(x[j],d)
            end
            push!(idx,(pos+1):(pos+size(x[n],d)))
        else
            push!(idx,1:size(y1,d))
        end
    end
    y2 = zeros(y1)
    y2[idx...] = x2
    return y2
end

@primitive  uncat(y1,n...),x2  uncat1(x2,y1,n...)
@primitive  uncat1(x2,y1,n...),y3  uncat(y3,n...)

# Here is a graphic that may explain the variable name choice where xi
# stands for the i'th order gradient:
#
# x  → cat    → y
#               ↓
# x1 ← uncat  ← y1
# ↓
# x2 → uncat1 → y2
#               ↓
# x3 ← uncat  ← y3

addtest(:cat, 1, 1., 2.)
addtest(:cat, 1, 1., [2.,3.])
addtest(:cat, 1, [1.,2.], 3.)
addtest(:cat, 1, [1.,2.], [3.,4.])
addtest(:cat, 1, [1. 2.], [3. 4.])
addtest(:cat, 2, 1., 2.)
addtest(:cat, 2, 1., [2. 3.])
addtest(:cat, 2, [1. 2.], 3.)
addtest(:cat, 2, [1.,2.], [3.,4.])
addtest(:cat, 2, [1. 2.], [3. 4.])

# vcat,hcat: should be defined in terms of cat. However base has some
# generic methods that prevent the cat call when the arguments are
# boxed.  This should fix it, at least when one of the first two args
# is boxed.
vcat(a::Rec,b::Rec,c...)=cat(1,a,b,c...)
vcat(a,b::Rec,c...)=cat(1,a,b,c...)
vcat(a::Rec,b...)=cat(1,a,b...)
hcat(a::Rec,b::Rec,c...)=cat(2,a,b,c...)
hcat(a,b::Rec,c...)=cat(2,a,b,c...)
hcat(a::Rec,b...)=cat(2,a,b...)

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
# hash: Works for Rec.
