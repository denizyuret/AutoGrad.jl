# Here we will define iteration (start,done,next) and indexing
# (getindex,setindex!,endof) interfaces for generic Value types.

# Julia handles access to AbstractArray, Associative, and Tuple
# subtypes using getindex:

# getindex(obj, key...) => value
# setindex!(obj, val, key...) => obj

# We do not allow overwriting, so setindex! for Values not allowed:

setindex!(x::Value,i...)=error("Overwriting operations currently not supported.")

# We handle the containers by overloading getindex:

@primitive  getindex(x,i...),dxi,xi  ungetindex(dxi,x,i...)
getindex{T<:Grad}(::Type{T},o...)=nothing # Only the first arg has gradient
fixdomain(::Fn{:getindex},i...)=(rand(2),1)

# Gradient of getindex: If xi=getindex(x,i...) and we receive dxi,
# ungetindex creates dx representing zeros similar to x, with only
# dx[i] set to dxi.  We use the sparse container OneHot for
# efficiency.

ungetindex(dxi,x,i...)=OneHot(dxi,x,i)

# For higher order derivatives, the operation of ungetindex might be
# recorded and differentiated.  We have:
# x -> getindex -> xi -> grad -> dxi -> ungetindex -> dx -> grad -> ddx -> getindex -> ddxi

@primitive ungetindex(dxi,x,i...),ddx,dx  getindex(ddx,i...)
ungetindex(::Type,::Value,dx...)=nothing # to avoid type ambiguity
ungetindex(::Type,::Any,dx...)=nothing   # to indicate no gradients except first
fixdomain(::Fn{:ungetindex},x...)=(rand(),rand(2),2)

# For efficiency we use the following sparse container

if !isdefined(:OneHot)
immutable OneHot; value; container; index; end
end

Base.sum(b::OneHot)=b.value
Base.getindex(b::OneHot,i...)=(if i==b.index; b.value; else; nothing; end)
Base.zeros(b::OneHot)=zeros(b.container)

function Base.full(b::OneHot)
    if isa(b.container,Tuple)
        ntuple(length(b.container)) do i
            if i==b.index[1]
                b.value
            else
                nothing
            end
        end
    else
        c=zeroslike(b.container)
        setindex!(c,b.value,b.index...)
        return c
    end
end

zeroslike{T<:Number}(a::AbstractArray{T})=zeros(a)
zeroslike(a::AbstractArray)=fill!(Array(Any,size(a)),nothing)
zeroslike(a::Associative)=similar(a)

_dbg(x::OneHot)="OH$(id2(x))_$(_dbg(x.container))_$((x.index...))_$(x.value)"
Base.show(io::IO, n::OneHot)= print(io, _dbg(n))

# sum_outgrads needs to handle OneHot values:
sum_outgrads(a::OneHot,b::OneHot)=(if a.index==b.index; OneHot(a.container,sum_outgrads(a.value,b.value),a.index); else; sum_outgrads(full(a),b); end)
sum_outgrads(a::Value,b::OneHot)=error((:sum,a,b))
sum_outgrads(a::OneHot,b::Value)=error((:sum,a,b))
sum_outgrads(a::Void,b::OneHot)=full(b)
sum_outgrads(a::OneHot,b::Void)=error((:sum,a,b))
sum_outgrads(a::OneHot,b)=error((:sum,a,Any))
sum_outgrads(a::Tuple,b::OneHot)=(ntuple(length(a)) do i; if i==b.index[1]; sum_outgrads(a[i],b.value); else; a[i]; end; end)
sum_outgrads(a::AbstractArray,b::OneHot)=setindex!(a,sum_outgrads(getindex(a,b.index...),b.value),b.index...)
sum_outgrads(a::Associative,b::OneHot)=setindex!(a,sum_outgrads(get(a,b.index...,nothing),b.value),b.index...)

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

for _f in interfacesNarg
    @eval @zerograd $_f(x,i...)
end

interfaces2arg = [
:(==),
:isequal,
:isless,
]                  

==(a::WeakRef,b::Value)=(a==b.value) # prevents clash with base.jl:68
==(a::Value,b::WeakRef)=(a.value==b) # prevents clash with base.jl:69

for _f in interfaces2arg
    @eval @zerograd $_f(a,b)
end

@primitive copy(x),dy dy
fixdomain(::Fn{:copy},x)=(rand(2),)

