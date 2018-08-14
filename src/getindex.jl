import Base: getindex, setindex!, sum, zeros, zero, ones, length, get

# Here we will define indexing (getindex,setindex!,firstindex,lastindex) 
# interface for generic Rec types.

# Julia handles access to AbstractArray, AbstractDict, and Tuple
# subtypes using getindex:

# getindex(obj, key...) => value
# setindex!(obj, val, key...) => obj

# We do not allow overwriting, so setindex! for Recs not allowed:

setindex!(x::Rec,v,I...)=error("Overwriting operations currently not supported.")

# We handle the containers by overloading getindex:

@primitive  getindex(x,i...),dxi,xi  ungetindex(x,dxi,i)
getindex(::Type{T},o...) where {T<:Grad} = nothing # Only the first arg has gradient

# For efficiency we use the following sparse container
# This object represents what you would get with
# setindex!(similar(container), value, index...)
# If there are repeated indices, the corresponding values should be summed.
# e.g. index could be an Int array

# TODO: we could have values/indices instead of value/index and use UngetIndex as a more efficient accumulator.
# TODO: use full much more rarely and keep things as UngetIndex. -- note: full is deprecated.
# TODO: implement KnetArray version of addindex! ???
# TODO: figure out julia4 problem with Array{CartesianIndex} -- j4 no longer supported.

struct UngetIndex; container; value; index; end

# Gradient of getindex: If xi=getindex(x,i...) and we receive dxi,
# ungetindex creates dx representing zeros similar to x, with only
# dx[i...] set to dxi.  We use the sparse container UngetIndex for
# efficiency.
# x -> getindex -> xi -> grad -> dxi -> ungetindex -> dx -> grad -> ddx -> getindex -> ddxi

ungetindex(x,dxi,i)=UngetIndex(x,dxi,i)

# For higher order derivatives, the operation of ungetindex might be
# recorded and differentiated, so it must be a primitive.  It is only
# differentiable wrt its value arg.  It should unbox its arguments,
# but it only needs to record if the value argument is boxed.  We'll
# have to define this manually.  To unbox the container arg and
# resolve ambiguity the ungetindex methods cover all combinations of
# first two args:
# (a,a), (a,r), (r,r), (r,a), (g2,a), (g2,r), (g,a), (g,r)

let ungetindex_r = recorder(ungetindex); global ungetindex
    ungetindex(x,dxi::Rec,i)=ungetindex_r(x,dxi,i)
end
ungetindex(x::Rec,dxi::Rec,i)=ungetindex(getval(x),dxi,getval(i))
ungetindex(x::Rec,dxi,i)=ungetindex(getval(x),dxi,getval(i))
ungetindex(::Type{Grad{2}},ddx,dx,x,dxi,i)=getindex(ddx,getval(i)...)
ungetindex(::Type{Grad{2}},ddx::Rec,dx,x,dxi,i)=getindex(ddx,getval(i)...)
ungetindex(::Type{T},o...) where {T<:Grad} = nothing
ungetindex(::Type{T},ddx::Rec,o...) where {T<:Grad} = nothing

# gradcheck works with the first arg, we need to check ungetindex grad for its second arg
# ungetindex2(value, container, index)=ungetindex(container, value, index)
# addtest(:ungetindex2, rand(),  rand(2),   (2,))
# addtest(:ungetindex2, rand(2), rand(3),   (2:3,))
# addtest(:ungetindex2, rand(),  rand(2,2), (1,2))
# addtest(:ungetindex2, rand(2), rand(3,3), (1:2,3))

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
    if length(b.index[1]) > 1
        cb = collect(Any,b.value)
    else
        cb = b.value
    end
    tuple(sum_outgrads_array(ca, cb, to_indices(ca,b.index)...)...)
end

# Dict has no multiple/repeated index problem, so simple setindex should work.
# If we change UngetIndex to have multiple indices, we need to be careful here.
function sum_outgrads(a::AbstractDict,b::UngetIndex)
    setindex!(a,sum_outgrads(get(a,b.index...,nothing),b.value),b.index...)
end

function sum_outgrads(a::AbstractArray,b::UngetIndex)
    # println((size(a),size(b.container),size(b.value),b.index))
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
        # @ncall $N setindex_shape_check x′ (d->idxlens[d]) # <-- different from _unsafe_setindex!
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

sum_outgrads_array(A::AbstractArray, X, I::CartesianIndex)=sum_outgrads_single(A,X,I)
sum_outgrads_array(A::AbstractArray, X, I::Real)=sum_outgrads_single(A,X,I)
sum_outgrads_array(A::AbstractArray, X, I::Colon)=sum_outgrads_single(A,X,I)
sum_outgrads_array(A::AbstractArray, X, I::AbstractArray{Bool})=sum_outgrads_single(A,X,I)
sum_outgrads_array(A::AbstractArray, X, I::AbstractRange)=sum_outgrads_single(A,X,I)
function sum_outgrads_single(A::AbstractArray, X, I)
    v = sum_outgrads(getindex(A,I), X)
    setindex!(A, v, I)
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
# sum_outgrads(a::Rec,b::UngetIndex)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b::Rec)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b::Nothing)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b)=error((:sum,a,Any))

sum(b::UngetIndex)=sum(b.value)
getindex(b::UngetIndex,i...)=getindex(full(b),i...) # TODO: solve without full(b)?
#zeros(b::UngetIndex)=zeros(b.container)             # TODO: solve without b.container?
zero(b::UngetIndex)=zero(b.container)               # TODO: solve without b.container?
ones(b::UngetIndex)=ones(b.container)
length(b::UngetIndex)=length(b.container)
full(b::UngetIndex)=sum_outgrads(zeroslike(b.container), b)

zeroslike(a::AbstractArray{T}) where {T<:Number} = zero(a)  # TODO: can this be nothing or an empty UngetIndex?
zeroslike(a::AbstractArray)=Array{Any}(nothing,size(a)) # TODO: can this be nothing or an empty UngetIndex?
zeroslike(a::AbstractDict)=empty(a)
zeroslike(a::Tuple)=ntuple(i->nothing, length(a))
zeroslike(o::UngetIndex)=zero(o) # TODO: can this be nothing or empty UngetIndex?
zeroslike(a::T) where {T<:Number} = T(0)   # This comes up if people use getindex on a single number

# get (getindex with a default value)
# This can be left as a composite function, it will get its gradient from getindex if necessary.
get(A::Rec{T}, i::Integer, default) where {T<:AbstractArray} = (if checkbounds(Bool, length(A), i); A[i]; else; default; end)
get(A::Rec{T}, I::Tuple{}, default) where {T<:AbstractArray} = similar(A, typeof(default), 0)
get(A::Rec{T}, I::Dims, default) where {T<:AbstractArray}    = (if checkbounds(Bool, size(A), I...); A[I...]; else; default; end)
