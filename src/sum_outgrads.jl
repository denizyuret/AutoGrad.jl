sum_outgrads(a::Number, b::Number)=a+b
sum_outgrads(a::Tuple, b::Tuple)=tuple([sum_outgrads(x,y) for (x,y) in zip(a,b)]...)
sum_outgrads(a::AbstractDict, b::AbstractDict) = (z=similar(a); for d in (a,b), (k,v) in d; z[k]=sum_outgrads(v,get(z,k,nothing)); end; z)
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

# sum_outgrads(accumulator,newval) needs to handle UngetElement values:

# The accumulator/outgrad values start as nothing.  They can
# potentially be returned to the user by gradfun, so the current
# design is to not use UngetElement for outgrad lest it gets exposed to
# the user. May rethink this from an efficiency perspective.

function sum_outgrads(a::Nothing,b::UngetElement)
    full(b) # TODO: do we need full here? consider keeping UngetElement as an accumulator.
end

function sum_outgrads(a::Tuple,b::UngetIndex)
    ca = collect(Any,a)
    if isa(fvalue(b),Tuple) # length(b.index[1]) > 1
        cb = collect(Any,fvalue(b))
    else
        cb = fvalue(b)
    end
    tuple(sum_outgrads_array(ca, cb, to_indices(ca,index(b))...)...)
end

# Dict has no multiple/repeated index problem, so simple setindex should work.
# If we change UngetIndex to have multiple indices, we need to be careful here.
function sum_outgrads(a::AbstractDict,b::UngetIndex)
    setindex!(a,sum_outgrads(get(a,index(b)...,nothing),fvalue(b)),index(b)...)
end

function sum_outgrads(a::AbstractArray,b::UngetIndex)
    # println((size(a),size(b.container),size(b.value),b.index))
    sum_outgrads_array(a, fvalue(b), to_indices(a,index(b))...)
end

sum_outgrads(a::Nothing, b::UngetProperty) = full(b) # fix ambiguity
function sum_outgrads(a, b::UngetProperty)
    @eval ($a..., $(index(b))=$(sum_outgrads(getproperty(a, index(b)), fvalue(b)))) # this might be slow
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
function sum_outgrads(a::T,b::T) where {T <: UngetElement}
    if index(a)==index(b)
        T(container(a),sum_outgrads(fvalue(a),fvalue(b)),index(a))
    else                        # TODO: we could always return UngetIndex if it supported multiple indices.
        sum_outgrads(full(a),b) # TODO: this can be erased if we use full above
    end
end

# This comes up if people use getindex on a number:
function sum_outgrads(a::T, b::UngetIndex) where {T<:Number}
    if !(index(b) == (1,) && isa(fvalue(b),T))
        throw(ArgumentError("sum_outgrads($a,$b)"))
    end
    a + fvalue(b)
end

# These should be never needed as long as we do not use UngetIndex as an accumulator on the LHS.
# sum_outgrads(a::Value,b::UngetIndex)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b::Value)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b::Nothing)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b)=error((:sum,a,Any))
