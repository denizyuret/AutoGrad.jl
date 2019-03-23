sum_outgrads(a::Number, b::Number)=a+b
sum_outgrads(a::Tuple, b::Tuple)=tuple([sum_outgrads(x,y) for (x,y) in zip(a,b)]...)
sum_outgrads(a::AbstractDict, b::AbstractDict) = (z=empty(a); for d in (a,b), (k,v) in d; z[k]=sum_outgrads(v,get(z,k,nothing)); end; z)
# We could have Array{Array} and Array{Any} added:
sum_outgrads(a::AbstractArray{T},b::AbstractArray) where T = (if isbitstype(T); (a+b); else; T[sum_outgrads(x,y) for (x,y) in zip(a,b)]; end)
# sum_outgrads needs to be a primitive for higher order gradients:
sum_outgrads(a::Value,b::Value)=forw(sum_outgrads,a,b)
sum_outgrads(a::Value,b)=forw(sum_outgrads,a,b)
sum_outgrads(a,b::Value)=forw(sum_outgrads,a,b)
back(::typeof(sum_outgrads),::Type{Arg{N}},dy,y,x1,x2) where N = dy
# we use `nothing` to indicate zero gradients
sum_outgrads(::Nothing,::Nothing)=nothing
sum_outgrads(a::Value,::Nothing)=a   # to avoid ambiguity
sum_outgrads(::Nothing,a::Value)=a   # to avoid ambiguity
sum_outgrads(a,::Nothing)=a
sum_outgrads(::Nothing,a)=a

# sum_outgrads(accumulator,newval) needs to handle UngetIndex values:

# The accumulator/outgrad values start as nothing.  They can
# potentially be returned to the user by gradfun, so the current
# design is to not use UngetIndex for outgrad lest it gets exposed to
# the user. May rethink this from an efficiency perspective.

function sum_outgrads(a::Nothing,b::UngetIndex)
    full(b) # TODO: do we need full here? consider keeping UngetIndex as an accumulator.
end

function sum_outgrads(a::Tuple,b::UngetIndex)
    ca = collect(Any,a)
    if isa(b.value,Tuple) # length(b.index[1]) > 1
        cb = collect(Any,b.value)
    else
        cb = b.value
    end
    tuple(sum_outgrads_array(ca, cb, to_indices(ca,b.index)...)...)
end

# Dict has no multiple/repeated index problem, so simple setindex should work.
# If we change UngetIndex to have multiple indices, we need to be careful here.
function sum_outgrads(a::AbstractDict,b::UngetIndex)
    if recording(); a = copy(a); end  # do not overwrite array if in highorder context
    setindex!(a,sum_outgrads(get(a,b.index...,nothing),b.value),b.index...)
end

function sum_outgrads(a::AbstractArray,b::UngetIndex)
    # println((size(a),size(b.container),size(b.value),b.index))
    if recording(); a = copy(a); end  # do not overwrite array if in highorder context
    sum_outgrads_array(a, b.value, to_indices(a,b.index)...)
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
sum_outgrads_array(A::AbstractArray, X, I::Union{Real,Colon,AbstractRange,AbstractArray{Bool}}...)=sum_outgrads_single(A,X,I...)
sum_outgrads_array(A::AbstractArray, X, I...)=sum_outgrads_single(A,X,I...)
function sum_outgrads_single(A::AbstractArray, X, I...)
    v = sum_outgrads(getindex(A,I...), X)
    setindex!(A, v, I...)
end

# This gets used in higher order gradients.
function sum_outgrads(a::UngetIndex,b::UngetIndex)
    if a.index==b.index
        UngetIndex(a.container,sum_outgrads(a.value,b.value),a.index)
    else                        # TODO: we could always return UngetIndex if it supported multiple indices.
        sum_outgrads(full(a),b) # TODO: this can be erased if we use full above
    end
end

# This comes up if people use getindex on a number:
function sum_outgrads(a::T, b::UngetIndex) where {T<:Number}
    if !(b.index == (1,) && isa(b.value,T))
        throw(ArgumentError("sum_outgrads($a,$b)"))
    end
    a + b.value
end

# These should be never needed as long as we do not use UngetIndex as an accumulator on the LHS.
# sum_outgrads(a::Value,b::UngetIndex)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b::Value)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b::Nothing)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b)=error((:sum,a,Any))
