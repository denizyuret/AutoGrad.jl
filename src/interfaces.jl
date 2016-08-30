# Here we will define iteration (start,done,next) and indexing
# (getindex,setindex!,endof) interfaces for generic Value types.

# Julia handles access to AbstractArray, Associative, and Tuple
# subtypes using getindex:

# getindex(obj, key...) => value
# setindex!(obj, val, key...) => obj

# So we can handle these container types by overloading getindex

@primitive  getindex(x,i...),dy  ungetindex(x,dy,i...)
setindex!(x::Value,i...)=error("Overwriting operations currently not supported.")
fixdomain(::Fn{:getindex},i...)=(rand(2),1)

# If y=getindex(x,i...) and we receive dy, ungetindex creates dx as
# zeros similar to x, with only dx[i] set to dy.  For efficiency zero
# arrays are represented with `nothing`. The user should define
# ungetindex for specific types.

ungetindex(x,a...)=throw(MethodError(ungetindex,x))
ungetindex{T}(x::AbstractArray{T}, dy, i...)=(dx = isbits(T) ? zeros(x) : fill!(Array(Any, size(x)), nothing); setindex!(dx,dy,i...); dx)
ungetindex(x::Associative, dy, i...) = (dx=similar(x);setindex!(dx,dy,i...);dx)
ungetindex(x::Tuple, dy, i) = ntuple(j->(j==i ? dy : nothing), length(x))

# In case of a higher order gradient, ungetindex would be called with
# Value inputs and needs to be recording primitive.
# dx=ungetindex(x,dy,i...) returned dx as all zeros with dx[i]=dy.
# Now we receive the gradient wrt its output: ddx.  To get the
# gradient wrt its dy input, we just need to extract ddx[i].  It is
# not differentiable wrt other inputs.

@primitive ungetindex(x::AbstractArray,dy,i...) # Need types to avoid ambiguity warnings
@primitive ungetindex(x::Associative,dy,i...)
@primitive ungetindex(x::Tuple,dy,i...)
fixdomain(::Fn{:ungetindex},x...)=(rand(2),rand(),1)
ungetindex(::Type{Grad{1}},ddx,dx,x,dy,i...) = 0
ungetindex(::Type{Grad{2}},ddx,dx,x,dy,i...) = getindex(ddx,i...)

# Iteration is used in `for x in a` loops and for `(x,y)=a` multiple
# assignments.

# start(a) => initial state
# done(a,state) => whether state s is final
# next(a,state) => (element, nextState)

# start and done return state and bool, not differentiable.
start(a::Value)=start(a.value)
done(a::Value,i)=done(a.value,i)

# next returns a tuple with an element and needs to be defined for each iterable.
# Specific types need to define their own next methods for Values:

next(a::Value,i)=throw(MethodError(next,(a,i)))
next{T<:Array}(a::Value{T},i) = (a[i],i+1)
next{T<:Tuple}(a::Value{T},i) = (a[i],i+1)
next{T<:Number}(a::Value{T},i) = (a,true)

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

for f in interfaces1arg
    @eval @zerograd $f(x)
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

for f in interfacesNarg
    @eval @zerograd $f(x,i...)
end

interfaces2arg = [
:(==),
:isequal,
:isless,
]                  

==(a::WeakRef,b::Value)=(a==b.value) # prevents clash with base.jl:68
==(a::Value,b::WeakRef)=(a.value==b) # prevents clash with base.jl:69

for f in interfaces2arg
    @eval @zerograd $f(a,b)
end

@primitive copy(x),dy dy
fixdomain(::Fn{:copy},x)=(rand(2),)

### DEAD CODE:

# Here are some examples for next, actual definitions are under base.
# next can be defined in terms of getindex, no need to make it a primitive.
# next{A<:AbstractArray}(a::Value{A},i)=(a[i],i+1)
# next{T<:Tuple}(a::Value{T},i)=(a[i],i+1)
# next{T<:Number}(a::Value{T},i)=(a,true)

# We no longer define getindex for each type as a primitive:
# import Base: getindex
# @primitive getindex{T<:Associative}(x::Value{T},i...)
# @primitive getindex{T<:Tuple}(x::Value{T},i)
# getindex(::D1,y,x,i...) = dy->ungetindex(x,dy,i...) 
# getindex{N}(::Dn{N},a...) = 0                 # only the first argument has a gradient

# ungetindex exampes:
# ungetindex(x::Tuple, dy, i)            = ntuple(j->(j==i ? dy : nothing), length(x))
# ungetindex(x::Associative, dy, i...)   = (dx=similar(x);setindex!(dx,dy,i...);dx)

# In case of a higher order gradient, ungetindex would be called with
# Value inputs and needs to be recorded.
# @primitive ungetindex{T<:Associative}(x::Value{T},dy,i...)
# @primitive ungetindex{T<:Tuple}(x::Value{T},dy,i)

# ungetindex{N}(::Dn{N}, dx...) = 0 # only arg 2 has a gradient

# getindex_r = recorder(getindex)
# getindex(x::Value,i...)=getindex_r(x,i...)
# getindex(::D1,y::Value,x::Value,i...) = dy->ungetindex(x,dy,i...) # y=x[i], dy=dJ/dy

