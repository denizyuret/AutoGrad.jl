import Base: getindex, setindex!, sum, zeros, zero, ones, length, get, 
             view, selectdim, getproperty, setproperty!

# Here we will define indexing (getindex,setindex!,firstindex,lastindex) 
# interface for generic Value types.

# Julia handles access to AbstractArray, AbstractDict, and Tuple
# subtypes using getindex:

# getindex(obj, key...) => value
# setindex!(obj, val, key...) => obj

function setindex!(x::Value,v,I...)
    if !isempty(_tapes)
        error("Array overwriting during gradient calculation not supported.")
    else
        setindex!(value(x),v,I...)
    end
end

function setproperty!(x::Value,i::Symbol,v)
    if !isempty(_tapes)
        error("Mutating objects during gradient calculation is not supported.")
    else
        setproperty!(value(x),i::Symbol,v)
    end
end

# We handle the containers by overloading getindex:

@primitive  getindex(x,i...),dxi,xi  ungetindex(x,dxi,i)
back(::typeof(getindex),::Type{Arg{N}},o...) where {N} = nothing # Only the first arg has gradient

@primitive  getproperty(x,i::Symbol),dxi,xi  ungetproperty(x,dxi,i)
back(::typeof(getproperty),::Type{Arg{N}},o...) where {N} = nothing # Only the first arg has gradient

# use ungetindex machinery also for view and selectdim
@primitive  view(x,i...),dxi,xi  ungetindex(x,dxi,i)
back(::typeof(view),::Type{Arg{N}},o...) where {N} = nothing # Only the first arg has gradient
@inline selectdim(A::Value{<:AbstractArray}, d::Integer, i) = Base._selectdim(A, d, i, Base.setindex(map(Base.Slice, axes(A)), i, d))

# For efficiency we use the following sparse container
# This object represents what you would get with
# setindex!(similar(container), value, index...)
# If there are repeated indices, the corresponding values should be summed.
# e.g. index could be an Int array

# TODO: we could have values/indices instead of value/index and use UngetIndex as a more efficient accumulator.
# TODO: use full much more rarely and keep things as UngetIndex. -- note: full is deprecated.
# TODO: implement KnetArray version of addindex! ???
# TODO: figure out julia4 problem with Array{CartesianIndex} -- j4 no longer supported.

struct UngetElement{GET, SET!}
    container
    value
    index
end

const UngetIndex = UngetElement{getindex, setindex!}
const UngetProperty = UngetElement{getproperty, setproperty!}

# field access functions since we overload getproperty for UngetElement
container(x::UngetElement) = getfield(x, :container)
# fvalue from core
index(x::UngetElement) = getfield(x, :index)

# Gradient of getindex: If xi=getindex(x,i...) and we receive dxi,
# ungetindex creates dx representing zeros similar to x, with only
# dx[i...] set to dxi.  We use the sparse container UngetIndex for
# efficiency.
# x -> getindex -> xi -> grad -> dxi -> ungetindex -> dx -> grad -> ddx -> getindex -> ddxi

ungetindex(x,dxi,i)=UngetIndex(x,dxi,i)
ungetproperty(x,dxi,i)=UngetProperty(x,dxi,i)

# For higher order derivatives, the operation of ungetindex might be
# recorded and differentiated, so it must be a primitive.  It is only
# differentiable wrt its value arg.  It should unbox its arguments,
# but it only needs to record if the value argument is boxed.  We'll
# have to define this manually.  To unbox the container arg and
# resolve ambiguity the ungetindex methods cover all combinations of
# first two args:
# (a,a), (a,r), (r,r), (r,a), (g2,a), (g2,r), (g,a), (g,r)

ungetindex(x,dxi::Value,i)=forw(ungetindex,x,dxi,i)
ungetindex(x::Value,dxi::Value,i)=ungetindex(value(x),dxi,value(i))
ungetindex(x::Value,dxi,i)=ungetindex(value(x),dxi,value(i))
back(::typeof(ungetindex),::Type{Arg{2}},ddx,dx,x,dxi,i)=getindex(ddx,value(i)...)
back(::typeof(ungetindex),::Type{Arg{N}},o...) where {N} = nothing

ungetproperty(x,dxi::Value,i)=forw(ungetproperty,x,dxi,i)
ungetproperty(x::Value,dxi::Value,i)=ungetproperty(value(x),dxi,value(i))
ungetproperty(x::Value,dxi,i)=ungetproperty(value(x),dxi,value(i))
back(::typeof(ungetproperty),::Type{Arg{2}},ddx,dx,x,dxi,i)=getproperty(ddx,value(i))
back(::typeof(ungetproperty),::Type{Arg{N}},o...) where {N} = nothing

# gradcheck works with the first arg, we need to check ungetindex grad for its second arg
# ungetindex2(value, container, index)=ungetindex(container, value, index)
# addtest(:ungetindex2, rand(),  rand(2),   (2,))
# addtest(:ungetindex2, rand(2), rand(3),   (2:3,))
# addtest(:ungetindex2, rand(),  rand(2,2), (1,2))
# addtest(:ungetindex2, rand(2), rand(3,3), (1:2,3))

sum(b::UngetElement)=sum(fvalue(b))
getindex(b::UngetElement,i...)=getindex(full(b),i...) # TODO: solve without full(b)?
getproperty(b::UngetProperty,i::Symbol)=getproperty(full(b),i) # TODO: solve without full(b)?
#zeros(b::UngetElement)=zeros(container(b))             # TODO: solve without container(b)?
zero(b::UngetElement)=zero(container(b))               # TODO: solve without container(b)?
ones(b::UngetElement)=ones(container(b))
length(b::UngetElement)=length(container(b))
full(b::UngetIndex)=sum_outgrads(zeroslike(container(b)), b)
full(b::UngetProperty)=sum_outgrads(zeroobject(container(b)), b)

zeroslike(a::AbstractArray{T}) where {T<:Number} = zero(a)  # TODO: can this be nothing or an empty UngetIndex?
zeroslike(a::AbstractArray)=Array{Any}(nothing,size(a)) # TODO: can this be nothing or an empty UngetIndex?
zeroslike(a::AbstractDict)=empty(a)
zeroslike(a::Tuple)=ntuple(i->nothing, length(a))
zeroslike(o::UngetIndex)=zero(o) # TODO: can this be nothing or empty UngetIndex?
zeroslike(a::T) where {T<:Number} = T(0)   # This comes up if people use getindex on a single number

zeroobject(a) = let p = propertynames(a)
    # use a named tuple which contains the same properties as the original object
    # why not use the original object?
    #   1. there is no `similar` for objects, and we don't know how to construct an arbitrary object
    #       - though it is possible to construct an object without calling it's constructor using Expr(:new)
    #   2. if the original object overloads `getproperty`, it's easily lead bug
    #       - for example, the original object may not even have enough fields to store these gradients
    # so the real choices are NamedTuples and Dicts. Not sure which is faster.
    NamedTuple{p}(ntuple(x->nothing, length(p)))
end

# get (getindex with a default value)
# This can be left as a composite function, it will get its gradient from getindex if necessary.
get(A::Value{T}, i::Integer, default) where {T<:AbstractArray} = (if checkbounds(Bool, length(A), i); A[i]; else; default; end)
get(A::Value{T}, I::Tuple{}, default) where {T<:AbstractArray} = similar(A, typeof(default), 0)
get(A::Value{T}, I::Dims, default) where {T<:AbstractArray}    = (if checkbounds(Bool, size(A), I...); A[I...]; else; default; end)
