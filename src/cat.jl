import Base: cat, vcat, hcat

# cat(A...; dims): Concatenate the input arrays along the specified
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
# cat(matrices...; dims=(1,2)) builds a block diagonal matrix, i.e. a
# block matrix with matrices[1], matrices[2], ... as diagonal blocks
# and matching zero blocks away from the diagonal.

# cat can take 0 or more arguments of any type.  In order to catch the
# cases where at least one arg is Value, we need to override the generic
# Base.cat(d,x...). This will call the recording cat_r if a Value is
# found, and the original _cat from the Base definition if not.

const NA = Union{Number,AbstractArray}
const NAR = Union{Number,AbstractArray,Value}

# Base has cat(x...; dims) defined, first specialize this:
cat(X::NA...; dims)=Base._cat(dims, X...) 

# Then define the method that catches at least one Value:
cat(X::NAR...; dims)=forw(cat,X...; dims=dims)
back(::typeof(cat),::Type{Arg{N}},y1,y,x...; dims) where {N}=uncat(y1,N,dims,x...)

# In Julia6+ dims can be Val{N} which breaks uncat:
uncat(y1,n,dims::Val{N},x...) where {N}=uncat(y1,n,N,x...)
uncat1(x2,y1,n,dims::Val{N},x...) where {N}=uncat1(x2,y1,n,N,x...)
# This resolves ambiguity with the @primitive version:
uncat(y1::Value,n,dims::Val{N},x...) where {N}=uncat(y1,n,N,x...)
uncat1(x2::Value,y1,n,dims::Val{N},x...) where {N}=uncat1(x2,y1,n,N,x...)

# For the gradient, we need to extract the n'th block from dy which
# has the same shape as y=cat(x...;dims).  Note that the inputs x[i]
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
    if isa(value(x[n]),Number)
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
    y2 = zero(y1)
    y2[idx...] = x2
    return y2
end

@primitive  uncat(y1,n,dims,x...),x2  uncat1(x2,y1,n,dims,x...)
@primitive  uncat1(x2,y1,n,dims,x...),y3  uncat(y3,n,dims,x...)

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

# cat1(x...)=cat(x...; dims=Val(1))
# cat2(x...)=cat(x...; dims=Val(2))
# addtestN(:cat1, 1., 2.)
# addtestN(:cat1, 1., [2.,3.])
# addtestN(:cat1, [1.,2.], 3.)
# addtestN(:cat1, [1.,2.], [3.,4.])
# addtestN(:cat1, [1. 2.], [3. 4.])
# addtestN(:cat2, 1., 2.)
# addtestN(:cat2, 1., [2. 3.])
# addtestN(:cat2, [1. 2.], 3.)
# addtestN(:cat2, [1.,2.], [3.,4.])
# addtestN(:cat2, [1. 2.], [3. 4.])

# vcat,hcat: should be defined in terms of cat. However base has some
# generic methods that prevent the cat call when all arguments are
# boxed.  This should fix it:

vcat(x::Value...) = cat(x...; dims=Val(1))
hcat(x::Value...) = cat(x...; dims=Val(2))

# vcat(a::Value,b::Value,c...)=cat(1,a,b,c...)
# vcat(a,b::Value,c...)=cat(1,a,b,c...)
# vcat(a::Value,b...)=cat(1,a,b...)
# hcat(a::Value,b::Value,c...)=cat(2,a,b,c...)
# hcat(a,b::Value,c...)=cat(2,a,b,c...)
# hcat(a::Value,b...)=cat(2,a,b...)

# The cat implementation above is slow when called with a lot of
# arguments (100s).  In our parser, the first epoch is a lot slower
# than the other epochs.  This is probably because there is a lot of
# compilation for different argument type patterns (where Values
# appear).  The following version tries to avoid all compilation at
# runtime.  It takes a bunch of arrays (not numbers), could be
# different shapes and sizes, and concatenates them as if they were
# all vectors.  It thus avoids a lot of shape related calculation as
# well.  It works with KnetArrays but slow, probably due to the copy
# kernel call overhead.  To make it fast with KnetArrays we need a
# single GPU kernel call which does all the copying.

"""
    cat1d(args...)

Return `vcat(vec.(args)...)` but possibly more efficiently. Can be used to concatenate the
contents of arrays with different shapes and sizes.
"""
function cat1d(args...)
    @inbounds for arg in args
        if isa(arg,Value)
            return forw(_cat1d, args...)
        end
    end
    return _cat1d(args...)
end

function _cat1d(args...)
    totlen = 0
    @inbounds for arg in args
        totlen += length(arg)
    end
    result = similar(args[1], totlen)
    offset1 = 1
    offset2 = 0
    @inbounds for arg in args
        offset2 += length(arg)
        result[offset1:offset2] = arg
        offset1 = offset2+1
    end
    return result
end

function back(::typeof(_cat1d),::Type{Arg{N}},y1,y,x...) where {N}
    offset2 = 0
    @inbounds for i=1:N
        offset2 += length(x[i])
    end
    offset1 = offset2 - length(x[N]) + 1
    reshape(y1[offset1:offset2], size(x[N]))
end

export cat1d

# addtestN(:cat1d, [1.,2.], [3.,4.])
# addtestN(:cat1d, [1. 2.], [3. 4.])
# addtestN(:cat1d, [1.,2.], [3. 4.])
# addtestN(:cat1d, [1. 2.], [3.,4.])
