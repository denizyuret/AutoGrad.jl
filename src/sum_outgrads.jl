"""
    sum_outgrads(accumulator, newval)

Add newval to accumulator and return the result. Used in outgrad calculations.  The outgrad
values start as `nothing` representing a 0 gradient. Then they are incremented using values
of type Number, Tuple, AbstractDict, AbstractArray, Nothing and AutoGrad.Sparse. The
accumulator and the newval types must match: Nothing matches all types, Sparse matches types
that match its container, other types must match themselves.
"""
function sum_outgrads end

# sum_outgrads needs to be a primitive for higher order gradients:
sum_outgrads(a::Value,b::Value)=forw(sum_outgrads,a,b)
sum_outgrads(a::Value,b)=forw(sum_outgrads,a,b)
sum_outgrads(a,b::Value)=forw(sum_outgrads,a,b)
back(::typeof(sum_outgrads),::Type{Arg{N}},dy,y,x1,x2) where N = dy

## Types with exact match
sum_outgrads(a::Number, b::Number)=a+b
sum_outgrads(a::Tuple, b::Tuple)=tuple([sum_outgrads(x,y) for (x,y) in zip(a,b)]...)
sum_outgrads(a::AbstractDict, b::AbstractDict) = (z=empty(a); for d in (a,b), (k,v) in d; z[k]=sum_outgrads(v,get(z,k,nothing)); end; z)
sum_outgrads(a::AbstractArray{T},b::AbstractArray) where T = (if isbitstype(T); (a+b); else; T[sum_outgrads(x,y) for (x,y) in zip(a,b)]; end)
# We could have Array{Array} and Array{Any} added, so no restriction on b.

## Nothing indicates zero gradient and matches any type
sum_outgrads(::Nothing,::Nothing)=nothing
sum_outgrads(a::Value,::Nothing)=a   # to avoid ambiguity
sum_outgrads(::Nothing,a::Value)=a   # to avoid ambiguity
sum_outgrads(a,::Nothing)=a
sum_outgrads(::Nothing,a)=a

## Sparse matches any type that matches its container
matches(::Any,::Any)=false
matches(::Number,::Number)=true
matches(::AbstractDict,::AbstractDict)=true
matches(a::Tuple,b::Tuple)=(length(a)===length(b))
matches(a::AbstractArray,b::AbstractArray)=(size(a)==size(b))

## If both accumulator and newval are sparse, merge:
function sum_outgrads(a::Sparse, b::Sparse)
    @assert matches(a.container, b.container) "$(summary.((a.container, b.container)))"
    Sparse(a.container, [ a.values; b.values ], [ a.indices; b.indices ])
end

## If sparse is the accumulator, reverse:
sum_outgrads(a::Sparse,b::Number)=sum_outgrads(b,a)
sum_outgrads(a::Sparse,b::Tuple)=sum_outgrads(b,a)
sum_outgrads(a::Sparse,b::AbstractDict)=sum_outgrads(b,a)
sum_outgrads(a::Sparse,b::AbstractArray)=sum_outgrads(b,a)

## Other types with Sparse:

# This comes up if people use getindex on a number:
function sum_outgrads(a::Number, b::Sparse)
    @assert matches(a, b.container)
    for (idx, val) in zip(b.indices, b.values)
        if !(idx == (1,) && isa(val,Number))
            throw(ArgumentError("sum_outgrads($a,$b)"))
        end
        a += val
    end
    return a
end

# Convert tuples to Any[] and use the array code
function sum_outgrads(a::Tuple,b::Sparse)
    @assert matches(a, b.container)
    ca = collect(Any,a)
    for (idx, val) in zip(b.indices, b.values)
        @assert length(idx) == 1
        cb = (idx[1] isa Real ? val : collect(Any,val))
        sum_outgrads_array(ca, cb, to_indices(ca,idx)...)
    end
    tuple(ca...)
end

# Dictionaries just increment for each index,value pair
function sum_outgrads(a::AbstractDict,b::Sparse)
    @assert matches(a, b.container)
    if recording(); a = copy(a); end  # do not overwrite array if in highorder context
    for (idx, val) in zip(b.indices, b.values)
        setindex!(a,sum_outgrads(get(a,idx...,nothing),val),idx...)
    end
    return a
end

# Need to be careful with arrays because of possible repeated indices
function sum_outgrads(a::AbstractArray,b::Sparse)
    @assert matches(a, b.container)
    if recording(); a = copy(a); end  # do not overwrite array if in highorder context
    for (idx, val) in zip(b.indices, b.values)
        sum_outgrads_array(a, val, to_indices(a,idx)...)
    end
    return a
end

# We need the following function to deal with repeated indices.
# Based on base/multidimensional.jl:634 _unsafe_setindex!
# Instead of last value overriding in case of repeated indices, we must sum.

using Base: unalias, index_lengths, setindex_shape_check
using Base.Cartesian # for @nexprs etc.

@generated function sum_outgrads_array(A::AbstractArray, x, I::Union{Real,AbstractArray}...)
    N = length(I)
    quote
        x′ = unalias(A, x)
        @nexprs $N d->(I_d = unalias(A, I[d]))
        idxlens = @ncall $N index_lengths I
        @ncall $N setindex_shape_check x′ (d->idxlens[d])
        Xy = iterate(x′)
        @inbounds @nloops $N i d->I_d begin
            # This is never reached, but serves as an assumption for
            # the optimizer that it does not need to emit error paths
            Xy === nothing && break
            (val, state) = Xy

            ai = @ncall $N getindex A i # <-- different from _unsafe_setindex!
            val = sum_outgrads(ai, val) # <-- different from _unsafe_setindex!

            @ncall $N setindex! A val i
            Xy = iterate(x′, state)
        end
        A
    end
end

# The following methods can assume there are no repeated indices:
# Only Array{Int} allows for repeated indices.
sum_outgrads_array(A::AbstractArray, X, I::Union{Real,Colon,AbstractRange,AbstractArray{Bool}}...) =
    sum_outgrads_single(A,X,I...)
sum_outgrads_array(A::AbstractArray, X, I...) =
    sum_outgrads_single(A,X,I...)
sum_outgrads_single(A::AbstractArray, X, I...) =
    setindex!(A, sum_outgrads(getindex(A,I...), X), I...)

