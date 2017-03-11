# Here we will define iteration (start,done,next) and indexing
# (getindex,setindex!,endof) interfaces for generic Rec types.

# Julia handles access to AbstractArray, Associative, and Tuple
# subtypes using getindex:

# getindex(obj, key...) => value
# setindex!(obj, val, key...) => obj

# We do not allow overwriting, so setindex! for Recs not allowed:

setindex!(x::Rec,v,i...)=error("Overwriting operations currently not supported.")

# We handle the containers by overloading getindex:

@primitive  getindex(x,i...),dxi,xi  ungetindex(x,dxi,i)
getindex{T<:Grad}(::Type{T},o...)=nothing # Only the first arg has gradient

# http://docs.julialang.org/en/latest/manual/arrays.html#man-supported-index-types-1
if VERSION < v"0.5.0"
    Base.IteratorsMD.CartesianIndex(i::Int...)=CartesianIndex(i)
end
addtest(getindex, rand(2), 1)   # Integer
addtest(getindex, rand(2,2), CartesianIndex(1,2)) # CartesianIndex
addtest(getindex, rand(3), [1,3]) # Vector{Int}
addtest(getindex, rand(4), [1 3; 2 4]) # Array{Int}
addtest(getindex, rand(2), []) # EmptyArray
addtest(getindex, rand(3), 2:3) # Range
addtest(getindex, rand(3), 1:2:3) # StridedRange
addtest(getindex, rand(2,2), [CartesianIndex(1,2),CartesianIndex(2,1)]) # Array{CartesianIndex}
addtest(getindex, rand(2,2), :, 1) # Colon
addtest(getindex, rand(3), [true, false, true]) # Array{Bool}
addtest(getindex, rand(2,2), 1, 2)
addtest(getindex, rand(3,3), 1:2, 3)
addtest(getindex, rand(3), [2,2]) # repeated index
addtest(getindex, rand(3,3), [2,2], :)
addtest(getindex, rand(3,3), :, [2,2])

# For efficiency we use the following sparse container
# This object represents what you would get with 
# setindex!(similar(container), value, index...)
# If there are repeated indices, the corresponding values should be summed.

# TODO: we could have values/indices instead of value/index and use UngetIndex as a more efficient accumulator.
# TODO: use full much more rarely and keep things as UngetIndex.
# TODO: fix full to use addindex!
# TODO: fix sum_outgrads as well to do the right thing.
# TODO: implement KnetArray version of addindex!
# TODO: figure out julia4 problem with Array{CartesianIndex}

immutable UngetIndex; container; value; index; end

# Gradient of getindex: If xi=getindex(x,i...) and we receive dxi,
# ungetindex creates dx representing zeros similar to x, with only
# dx[i...] set to dxi.  We use the sparse container UngetIndex for
# efficiency.
# x -> getindex -> xi -> grad -> dxi -> ungetindex -> dx -> grad -> ddx -> getindex -> ddxi

ungetindex(x,dxi,i)=UngetIndex(x,dxi,i)

# For higher order derivatives, the operation of ungetindex might be
# recorded and differentiated, so it must be a primitive.  It is only
# differentiable wrt its value arg.  The following methods cover
# (a,a), (a,r), (g,a), (g2,a) argtypes.

ungetindex_r = recorder(ungetindex)
ungetindex(x,dxi::Rec,i)=ungetindex_r(x,dxi,i)
ungetindex(::Type{Grad{2}},ddx,dx,x,dxi,i)=getindex(ddx,getval(i)...)
ungetindex(::Type,o...)=nothing

# It should unbox its arguments, but it only needs to record if the
# value argument is boxed.  We'll have to define this manually.  To
# unbox the container arg and resolve ambiguity the following methods
# cover (r,r), (r,a), (g,r), (g2,r).

ungetindex(x::Rec,dxi::Rec,i)=ungetindex(getval(x),dxi,getval(i))
ungetindex(x::Rec,dxi,i)=ungetindex(getval(x),dxi,getval(i))
ungetindex(::Type{Grad{2}},ddx::Rec,dx,x,dxi,i)=getindex(ddx,getval(i)...)
ungetindex(::Type,ddx::Rec,o...)=nothing

addtest(ungetindex, rand(2),   rand(),  (2,))
addtest(ungetindex, rand(3),   rand(2), (2:3,))
addtest(ungetindex, rand(2,2), rand(),  (1,2))
addtest(ungetindex, rand(3,3), rand(2), (1:2,3))

Base.sum(b::UngetIndex)=sum(b.value)
Base.getindex(b::UngetIndex,i...)=getindex(full(b),i...) # TODO: solve without full(b)?
Base.zeros(b::UngetIndex)=zeros(b.container)             # TODO: solve without b.container?
Base.ones(b::UngetIndex)=ones(b.container)
Base.length(b::UngetIndex)=length(b.container)

function Base.full(b::UngetIndex)
    value = b.value
    index = b.index[1]
    if isa(value,Tuple); value = collect(value); end
    # If index is an array of Int's or CartesianIndex{N}'s, values for repeated indices need to be summed
    if length(b.index)==1 && isa(index,Array) && !isa(index,Array{Bool}) && length(index) > 1 && !allunique(index)
        vdict = Dict{eltype(index),eltype(value)}()
        for i in 1:length(index)
            setindex!(vdict, value[i]+get(vdict,index[i],zero(value[i])), index[i])
        end
        for i in 1:length(index)
            value[i] = vdict[index[i]]
        end
    end
    if isa(b.container,Tuple); c = zeroslike(collect(b.container)); else; c = zeroslike(b.container); end
    setindex!(c, value, b.index...)
    if isa(b.container,Tuple); c = tuple(c...); end
    return c
end

zeroslike{T<:Number}(a::AbstractArray{T})=zeros(a)  # TODO: can this be nothing or an empty UngetIndex?
zeroslike(a::AbstractArray)=fill!(Array(Any,size(a)),nothing) # TODO: can this be nothing or an empty UngetIndex?
zeroslike(a::Associative)=similar(a)
zeroslike(o::UngetIndex)=zeros(o) # TODO: can this be nothing or empty UngetIndex?

_dbg(x::UngetIndex)="U$(id2(x))_$(_dbg(x.container))_$(_dbg(x.value))_$((x.index...))"
Base.show(io::IO, n::UngetIndex)= print(io, _dbg(n))

# sum_outgrads needs to handle UngetIndex values:
sum_outgrads(a::UngetIndex,b::UngetIndex)=(if a.index==b.index; UngetIndex(a.container,sum_outgrads(a.value,b.value),a.index); else; sum_outgrads(full(a),b); end)
sum_outgrads(a::Rec,b::UngetIndex)=error((:sum,a,b))
sum_outgrads(a::UngetIndex,b::Rec)=error((:sum,a,b))
sum_outgrads(a::Void,b::UngetIndex)=full(b) # TODO: do we need full here?
sum_outgrads(a::UngetIndex,b::Void)=error((:sum,a,b))
sum_outgrads(a::UngetIndex,b)=error((:sum,a,Any))
sum_outgrads(a::Tuple,b::UngetIndex)=(b=full(b);ntuple(length(a)) do i; sum_outgrads(a[i],b[i]); end) # TODO: do we need full here?
sum_outgrads(a::AbstractArray,b::UngetIndex)=setindex!(a,sum_outgrads(getindex(a,b.index...),b.value),b.index...) # TODO: fix repeated index bug
sum_outgrads(a::Associative,b::UngetIndex)=setindex!(a,sum_outgrads(get(a,b.index...,nothing),b.value),b.index...)


### ITERATION

# Iteration is used in `for x in a` loops and for `(x,y)=a` multiple
# assignments.

# start(a) => initial state
# done(a,state) => whether state s is final
# next(a,state) => (element, nextState)

# start and done return state and bool, not differentiable.
start(a::Rec)=start(a.value)
done(a::Rec,i)=done(a.value,i)

# next returns a tuple with an element and needs to be defined for each iterable.
# Specific types need to define their own next methods for Recs:

next(a::Rec,i)=throw(MethodError(next,(a,i)))
next{T<:Array}(a::Rec{T},i) = (a[i],i+1)
next{T<:Tuple}(a::Rec{T},i) = (a[i],i+1)
next{T<:Number}(a::Rec{T},i) = (a,true)
# This needs more work:
# next{T<:Base.RecIterator}(a::Rec{T},i) = (d=a.value.dict; (d.vals[i], skip_deleted(d,i+1)))

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

if VERSION >= v"0.5.0"
# to prevent ambiguity with abstractarray.jl:470
@zerograd similar(x, dims::Base.DimOrInd...) 
end

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
addtest(copy, rand(2))
