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
# checkbounds: interfaces.jl
# throw_boundserror: Not exported
# _internal_checkbounds: Not exported
# similar: interfaces.jl
# reshape
@primitive reshape(x,i...),dy  reshape(dy,size(x))
addtest(:reshape,rand(2,2),(4,1))
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

# After dims, cat can take 0 or more arguments of any type.  In order
# to catch the cases where at least one arg is Rec, we need to
# override the generic Base.cat(d,x...). This will call the recording
# cat_r if a Rec is found, and the original cat_t from the Base
# definition if not.

const NA = Union{Number,AbstractArray}
const NAR = Union{Number,AbstractArray,Rec}

# Base has cat(dims,x...) defined, first specialize this:
cat(dims, X::NA...)=Base.cat_t(dims, prom_(X...), X...)

# Then define the method that catches at least one Rec:
cat_r = recorder(cat)
cat(dims, X::NAR...)=cat_r(dims, X...)

cat(::Type{Grad{1}},a::AbstractArray...)=nothing # julia4 ambiguity fix
cat(::Type{Grad{1}},a::NA...)=nothing # ambiguity fix
cat(::Type{Grad{1}},a::NAR...)=nothing # ambiguity fix
cat(::Type{Grad{1}},a...)=nothing

cat{N}(::Type{Grad{N}},y1::AbstractArray,y::AbstractArray,dims::AbstractArray,x::AbstractArray...)=uncat(y1,N-1,dims,x...)   # ambiguity fix
cat{N}(::Type{Grad{N}},y1::NA,y::NA,dims::NA,x::NA...)=uncat(y1,N-1,dims,x...)   # ambiguity fix
cat{N}(::Type{Grad{N}},y1::NAR,y::NAR,dims::NAR,x::NAR...)=uncat(y1,N-1,dims,x...)   # ambiguity fix
cat{N}(::Type{Grad{N}},y1,y,dims,x...)=uncat(y1,N-1,dims,x...)   # N-1 because first arg is catdims
prom_(X...) = Base.promote_eltypeof(X...)

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

# In Julia6+ dims can be Val{N} which breaks uncat:
uncat{N}(y1,n,dims::Type{Val{N}},x...)=uncat(y1,n,N,x...)
uncat1{N}(x2,y1,n,dims::Type{Val{N}},x...)=uncat1(x2,y1,n,N,x...)

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

cat1(x...)=cat(1,x...)
cat2(x...)=cat(2,x...)
addtestN(:cat1, 1., 2.)
addtestN(:cat1, 1., [2.,3.])
addtestN(:cat1, [1.,2.], 3.)
addtestN(:cat1, [1.,2.], [3.,4.])
addtestN(:cat1, [1. 2.], [3. 4.])
addtestN(:cat2, 1., 2.)
addtestN(:cat2, 1., [2. 3.])
addtestN(:cat2, [1. 2.], 3.)
addtestN(:cat2, [1.,2.], [3.,4.])
addtestN(:cat2, [1. 2.], [3. 4.])

# vcat,hcat: should be defined in terms of cat. However base has some
# generic methods that prevent the cat call when all arguments are
# boxed.  This should fix it:

vcat(x::Rec...) = cat(1, x...)
hcat(x::Rec...) = cat(2, x...)

# vcat(a::Rec,b::Rec,c...)=cat(1,a,b,c...)
# vcat(a,b::Rec,c...)=cat(1,a,b,c...)
# vcat(a::Rec,b...)=cat(1,a,b...)
# hcat(a::Rec,b::Rec,c...)=cat(2,a,b,c...)
# hcat(a,b::Rec,c...)=cat(2,a,b,c...)
# hcat(a::Rec,b...)=cat(2,a,b...)

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

# The cat implementation above is slow when called with a lot of
# arguments (100s).  In our parser, the first epoch is a lot slower
# than the other epochs.  This is probably because there is a lot of
# compilation for different argument type patterns (where Recs
# appear).  The following version tries to avoid all compilation at
# runtime.  It takes a bunch of arrays (not numbers), could be
# different shapes and sizes, and concatenates them as if they were
# all vectors.  It thus avoids a lot of shape related calculation as
# well.  It works with KnetArrays but slow, probably due to the copy
# kernel call overhead.  To make it fast with KnetArrays we need a
# single GPU kernel call which does all the copying.

gradarg{N}(::Type{Grad{N}})=N

function cat1d(g::DataType,y1,y,x...) # g = Grad{N}
    argnum = gradarg(g)
    offset2 = 0
    @inbounds for i=1:argnum
        offset2 += length(x[i])
    end
    offset1 = offset2 - length(x[argnum]) + 1
    reshape(y1[offset1:offset2], size(x[argnum]))
end

function cat1d(args...)
    totlen = 0
    @inbounds for arg in args
        totlen += length(arg)
    end
    result = similar(getval(args[1]), totlen)
    offset1 = 1
    offset2 = 0
    @inbounds for arg in args
        offset2 += length(arg)
        result[offset1:offset2] = getval(arg)
        offset1 = offset2+1
    end
    @inbounds for argnum = 1:length(args)
        arg = args[argnum]
        if !isa(arg,Rec); continue; end
        for t=1:length(arg.tapes)
            tape = arg.tapes[t]
            if iscomplete(tape); continue; end
            parent = arg.nodes[t]
            if !isa(result,Rec) 
                result = Rec(result, tape; func=cat1d, args=args)
                rnode = result.nodes[1]
            else
                s = findeq(result.tapes, tape)
                if s > 0
                    rnode = result.nodes[s]
                else
                    rnode = Node(result, tape)
                end
            end
            rnode.parents[argnum] = parent
        end
    end
    return result
end

export cat1d

addtestN(:cat1d, [1.,2.], [3.,4.])
addtestN(:cat1d, [1. 2.], [3. 4.])
addtestN(:cat1d, [1.,2.], [3. 4.])
addtestN(:cat1d, [1. 2.], [3.,4.])
