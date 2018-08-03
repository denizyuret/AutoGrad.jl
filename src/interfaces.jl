import Base: ==, checkbounds, copy, done, eachindex, eltype, endof, full, getindex, indexed_iterate, isassigned, isempty, isequal, isless, iterate, length, ndims, next, one, ones, pointer, setindex!, show, similar, size, start, stride, strides, sum, zero, zeros

# Here we will define iteration (start,done,next) and indexing
# (getindex,setindex!,endof) interfaces for generic Rec types.

# Julia handles access to AbstractArray, AbstractDict, and Tuple
# subtypes using getindex:

# getindex(obj, key...) => value
# setindex!(obj, val, key...) => obj

# We do not allow overwriting, so setindex! for Recs not allowed:

setindex!(x::Rec,v,i...)=error("Overwriting operations currently not supported.")

# We handle the containers by overloading getindex:

@primitive  getindex(x,i...),dxi,xi  ungetindex(x,dxi,i)
getindex(::Type{T},o...) where {T<:Grad} = nothing # Only the first arg has gradient

# For efficiency we use the following sparse container
# This object represents what you would get with
# setindex!(similar(container), value, index...)
# If there are repeated indices, the corresponding values should be summed.

# TODO: we could have values/indices instead of value/index and use UngetIndex as a more efficient accumulator.
# TODO: use full much more rarely and keep things as UngetIndex.
# TODO: implement KnetArray version of addindex!
# TODO: figure out julia4 problem with Array{CartesianIndex}

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
ungetindex2(value, container, index)=ungetindex(container, value, index)
addtest(:ungetindex2, rand(),  rand(2),   (2,))
addtest(:ungetindex2, rand(2), rand(3),   (2:3,))
addtest(:ungetindex2, rand(),  rand(2,2), (1,2))
addtest(:ungetindex2, rand(2), rand(3,3), (1:2,3))

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
    tuple(sum_outgrads_array(ca, cb, b.index...)...)
end

# Dict has no multiple/repeated index problem, so simple setindex should work.
# If we change UngetIndex to have multiple indices, we need to be careful here.
function sum_outgrads(a::AbstractDict,b::UngetIndex)
    setindex!(a,sum_outgrads(get(a,b.index...,nothing),b.value),b.index...)
end

function sum_outgrads(a::AbstractArray,b::UngetIndex)
    # println((size(a),size(b.container),size(b.value),b.index))
    sum_outgrads_array(a, b.value, b.index...)
end

# We need the following two functions to deal with repeated indices.

# Based on base/multidimensional.jl:420 _unsafe_batchsetindex!
# using Base: index_lengths, setindex_shape_check, decolon # these are not portable!
using Base.Cartesian

@generated function sum_outgrads_array(A::AbstractArray, X, I::Union{Real,AbstractArray,Colon}...)
    N = length(I)
    quote
        ### We need to handle bool arrays and colons here
        @nexprs $N d->(I_d = I[d]; if isa(I_d,AbstractArray{Bool}); I_d=findall(I_d); elseif isa(I_d,Colon); I_d=1:size(A,d); end)
        ### Using nothing for zero array fails this check
        # idxlens = @ncall $N index_lengths A I
        # @ncall $N setindex_shape_check X (d->idxlens[d])
        ### julia4 does not have decolon
        # J = @ncall $N decolon A I
        # @nexprs $N d->(J_d = J[d])
        Xs = start(X)
        @inbounds @nloops $N i d->I_d begin
            v, Xs = next(X, Xs)
            u = @ncall $N getindex A i
            w = sum_outgrads(u,v)
            @ncall $N setindex! A w i
        end
        A
    end
end

function sum_outgrads_array(A::AbstractArray, X, I::AbstractArray)
    Xs = start(X)
    @inbounds for i in I
        v, Xs = next(X, Xs)
        u = getindex(A, i)
        w = sum_outgrads(u,v)
        setindex!(A,w,i)
    end
    A
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
zeros(b::UngetIndex)=zeros(b.container)             # TODO: solve without b.container?
zero(b::UngetIndex)=zero(b.container)               # TODO: solve without b.container?
ones(b::UngetIndex)=ones(b.container)
length(b::UngetIndex)=length(b.container)
full(b::UngetIndex)=sum_outgrads(zeroslike(b.container), b)

zeroslike(a::AbstractArray{T}) where {T<:Number} = zero(a)  # TODO: can this be nothing or an empty UngetIndex?
zeroslike(a::AbstractArray)=Array{Any}(nothing,size(a)) # TODO: can this be nothing or an empty UngetIndex?
zeroslike(a::AbstractDict)=empty(a)
zeroslike(a::Tuple)=ntuple(i->nothing, length(a))
zeroslike(o::UngetIndex)=zeros(o) # TODO: can this be nothing or empty UngetIndex?
zeroslike(a::T) where {T<:Number} = T(0)   # This comes up if people use getindex on a single number

_dbg(x::UngetIndex)="U$(id2(x))_$(_dbg(x.container))_$(_dbg(x.value))_$((x.index...,))"
show(io::IO, n::UngetIndex)= print(io, _dbg(n))

### ITERATION: We define iterate in terms of getindex which does the necessary tracking.

# Iterate is used in `for x in a` loops 
iterate(a::Rec,i=nothing)=throw(MethodError(iterate,(a,i)))
iterate(a::Rec{T},i=1) where {T<:AbstractArray} = i > length(a) ? nothing : (a[i],i+1)
iterate(a::Rec{T},i=1) where {T<:Tuple}         = i > length(a) ? nothing : (a[i],i+1)
iterate(a::Rec{T})   where {T<:AbstractDict} = (v=iterate(a.value);   v==nothing ? v : (((k,v),j)=v;(k=>a[k],j)))
iterate(a::Rec{T},i) where {T<:AbstractDict} = (v=iterate(a.value,i); v==nothing ? v : (((k,v),j)=v;(k=>a[k],j)))
iterate(a::Rec{T})   where {T<:Number} = (a,nothing)
iterate(a::Rec{T},i) where {T<:Number} = nothing

# indexed_iterate for `(x,y)=a` multiple assignments.
indexed_iterate(a::Rec{T},i::Int,state=1) where {T<:Tuple} = (a[i],i+1)
indexed_iterate(a::Rec{T},i::Int,state=1) where {T<:Array} = (a[i],i+1)

# Finally here are some common functions that do not return floats
# (e.g. length) or return constant outputs (e.g. zero).

interfaces1arg = [
:eltype,
:endof,
:isempty,
:length,
:ndims,
:one,
:ones,
:strides,
:zero,
:zeros,
]

for _f in interfaces1arg
    @eval @zerograd $_f(x)
end

interfaces1arg_type = [
:eltype,
:ndims,
:one,
:zero,
]

for _f in interfaces1arg_type
    @eval $_f(::Type{Rec{T}}) where T = $_f(T)
end

interfacesNarg = [
:checkbounds,
:eachindex,
:isassigned,
:pointer,
:similar,
:size,
:stride,
]

@zerograd similar(x)
@zerograd similar(f, shape::Tuple) # abstractarray.jl:565
@zerograd similar(x, dims::Base.DimOrInd...) # abstractarray.jl:566

for _f in interfacesNarg
    @eval @zerograd $_f(x,i...)
end

interfaces2arg = [
:(==),
:isequal,
:isless,
]

==(a::WeakRef,b::Rec)=(a==b.value) # prevents clash with base.jl:68
==(a::Rec,b::WeakRef)=(a.value==b) # prevents clash with base.jl:69

for _f in interfaces2arg
    @eval @zerograd $_f(a,b)
end

@primitive copy(x),dy dy
addtest(:copy, rand(2))

# issue #18
size(a::Rec, d1::Integer, d2::Integer, dx::Vararg{Integer}) = size(getval(a), d1, d2, dx...)
