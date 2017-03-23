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

# For efficiency we use the following sparse container
# This object represents what you would get with 
# setindex!(similar(container), value, index...)
# If there are repeated indices, the corresponding values should be summed.

# TODO: we could have values/indices instead of value/index and use UngetIndex as a more efficient accumulator.
# TODO: use full much more rarely and keep things as UngetIndex.
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
ungetindex{T<:Grad}(::Type{T},o...)=nothing
ungetindex{T<:Grad}(::Type{T},ddx::Rec,o...)=nothing

addtest(ungetindex, rand(2),   rand(),  (2,))
addtest(ungetindex, rand(3),   rand(2), (2:3,))
addtest(ungetindex, rand(2,2), rand(),  (1,2))
addtest(ungetindex, rand(3,3), rand(2), (1:2,3))

# sum_outgrads(accumulator,newval) needs to handle UngetIndex values:

# The accumulator/outgrad values start as nothing.  They can
# potentially be returned to the user by gradfun, so the current
# design is to not use UngetIndex for outgrad lest it gets exposed to
# the user. May rethink this from an efficiency perspective.

function sum_outgrads(a::Void,b::UngetIndex)
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
# TODO: this overwrites a, will not work with higher level gradients
function sum_outgrads(a::Associative,b::UngetIndex)
    setindex!(a,sum_outgrads(get(a,b.index...,nothing),b.value),b.index...)
end

# TODO: this overwrites a, will not work with higher level gradients
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
        @nexprs $N d->(I_d = I[d]; if isa(I_d,AbstractArray{Bool}); I_d=find(I_d); elseif isa(I_d,Colon); I_d=1:size(A,d); end)
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
sum_outgrads_array(A::AbstractArray, X, I::Range)=sum_outgrads_single(A,X,I)
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
function sum_outgrads{T<:Number}(a::T, b::UngetIndex)
    if !(b.index == (1,) && isa(b.value,T))
        throw(ArgumentError("sum_outgrads($a,$b)"))
    end
    a + b.value
end

# These should be never needed as long as we do not use UngetIndex as an accumulator on the LHS.
# sum_outgrads(a::Rec,b::UngetIndex)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b::Rec)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b::Void)=error((:sum,a,b))
# sum_outgrads(a::UngetIndex,b)=error((:sum,a,Any))

sum(b::UngetIndex)=sum(b.value)
getindex(b::UngetIndex,i...)=getindex(full(b),i...) # TODO: solve without full(b)?
zeros(b::UngetIndex)=zeros(b.container)             # TODO: solve without b.container?
ones(b::UngetIndex)=ones(b.container)
length(b::UngetIndex)=length(b.container)
full(b::UngetIndex)=sum_outgrads(zeroslike(b.container), b)

zeroslike{T<:Number}(a::AbstractArray{T})=zeros(a)  # TODO: can this be nothing or an empty UngetIndex?
zeroslike(a::AbstractArray)=fill!(Array(Any,size(a)),nothing) # TODO: can this be nothing or an empty UngetIndex?
zeroslike(a::Associative)=similar(a)
zeroslike(a::Tuple)=ntuple(i->nothing, length(a))
zeroslike(o::UngetIndex)=zeros(o) # TODO: can this be nothing or empty UngetIndex?
zeroslike{T<:Number}(a::T)=T(0)   # This comes up if people use getindex on a single number

_dbg(x::UngetIndex)="U$(id2(x))_$(_dbg(x.container))_$(_dbg(x.value))_$((x.index...))"
Base.show(io::IO, n::UngetIndex)= print(io, _dbg(n))

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
