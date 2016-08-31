# Here we will define iteration (start,done,next) and indexing
# (getindex,setindex!,endof) interfaces for generic Value types.

# Julia handles access to AbstractArray, Associative, and Tuple
# subtypes using getindex:

# getindex(obj, key...) => value
# setindex!(obj, val, key...) => obj

# We do not allow overwriting:

setindex!(x::Value,i...)=error("Overwriting operations currently not supported.")

# We handle these container types by overloading getindex

@primitive  getindex(x,i...),dxi,xi  ungetindex(dxi,x,i...)
getindex{N}(::Type{Grad{N}},o...)=nothing
fixdomain(::Fn{:getindex},i...)=(rand(2),1)

# If xi=getindex(x,i...) and we receive dxi, ungetindex creates dx
# representing zeros similar to x, with only dx[i] set to dxi.  We use
# the sparse container OneHot for efficiency.

ungetindex(dxi,x,i...)=OneHot(dxi,x,i)

# For higher order derivatives, the operation of ungetindex might be
# recorded and differentiated.

@primitive ungetindex(dxi,x,i...),ddx,dx  getindex(ddx,i...)
ungetindex{N}(::Type{Grad{N}},::Value,o...)=nothing

# x -> getindex -> xi -> grad -> dxi -> ungetindex -> dx -> grad -> ddx -> getindex -> ddxi

#@primitive ungetindex(x,dxi,i...),ddx,dx  nothing  getindex(ddx,i...)
#@primitive ungetindex(x,dxi,i...),ddx,dx (println((:ug1,nothing));nothing) (ddxi=getindex(ddx,i...);println((:ug2,ddxi,x,dxi,i...));ddxi)
# ungetindex_r = recorder(ungetindex)
# ungetindex(x::Value,dxi::Value,i...)=ungetindex_r(x,dxi,i...)
# ungetindex(::Type{Grad{2}},ddx,dx,x,dxi,i...)=getindex(ddx,i...)
# ungetindex{N}(::Type{Grad{N}},o...)=nothing
fixdomain(::Fn{:ungetindex},x...)=(rand(),rand(2),2)

# For efficiency we use the following sparse container

immutable OneHot{T}; value; container::T; index; end

Base.sum(b::OneHot)=b.value
Base.full(b::OneHot)=(c=zeroslike(b.container);setindex!(c,b.value,b.index...);c)
Base.full{T<:Tuple}(b::OneHot{T})=(ntuple(length(b.container)) do i; if i==b.index[1]; b.value; else; nothing; end; end)
Base.getindex(b::OneHot,i...)=(if i==b.index; b.value; else; nothing; end)
zeroslike{T<:Number}(a::AbstractArray{T})=zeros(a)
zeroslike(a::AbstractArray)=fill!(similar(a),nothing)
zeroslike(a::Associative)=similar(a)
_dbg(x::OneHot)="OH$(id2(x))_$(id2(x.container))_$((x.index...))_$(x.value)"
Base.show(io::IO, n::OneHot)= print(io, _dbg(n))

# sum_outgrads needs to handle OneHot values:
sum_outgrads(a::OneHot,b::OneHot)=(if a.index==b.index; OneHot(a.container,sum_outgrads(a.value,b.value),a.index); else; sum_outgrads(full(a),b); end)
sum_outgrads(a::Value,b::OneHot)=error((:sum,a,b))
sum_outgrads(a::OneHot,b::Value)=error((:sum,a,b))
sum_outgrads(a::Void,b::OneHot)=full(b)
sum_outgrads(a::OneHot,b::Void)=error((:sum,a,b))
sum_outgrads(a::OneHot,b)=error((:sum,a,Any))
sum_outgrads(a,b::OneHot)=setindex!(a,sum_outgrads(get(a,b.index,nothing),b.value),b.index...)
sum_outgrads(a::Tuple,b::OneHot)=(ntuple(length(a)) do i; if i==b.index[1]; sum_outgrads(a[i],b.value); else; a[i]; end; end)


# ungetindex(x,a...)=throw(MethodError(ungetindex,x))
# ungetindex{T}(x::AbstractArray{T}, dy, i...)=(dx = isbits(T) ? zeros(x) : fill!(Array(Any, size(x)), nothing); setindex!(dx,dy,i...); dx)
# ungetindex(x::Associative, dy, i...) = (dx=similar(x);setindex!(dx,dy,i...);dx)
# ungetindex(x::Tuple, dy, i) = ntuple(j->(j==i ? dy : nothing), length(x))

# In case of a higher order gradient, ungetindex would be called with
# Value inputs and needs to be recording primitive.
# dx=ungetindex(x,dy,i...) returned dx as all zeros with dx[i]=dy.
# Now we receive the gradient wrt its output: ddx.  To get the
# gradient wrt its dy input, we just need to extract ddx[i].  It is
# not differentiable wrt other inputs.

# @primitive ungetindex(x::AbstractArray,dy,i...) # Need types to avoid ambiguity warnings
# @primitive ungetindex(x::Associative,dy,i...)
# @primitive ungetindex(x::Tuple,dy,i...)
# ungetindex(::Type{Grad{1}},ddx,dx,x,dy,i...) = 0
# ungetindex(::Type{Grad{2}},ddx,dx,x,dy,i...) = getindex(ddx,i...)

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

