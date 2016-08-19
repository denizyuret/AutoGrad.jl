# Let's try to handle iteration for arrays and tuples: This is useful
# in `for x in a` loops and for `(x,y)=a` multiple assignments.

# start(a) => initial state
# done(a,state) => whether state s is final
# next(a,state) => (element, nextState)

# start and done return state and bool, not differentiable.
# next can be defined in terms of getindex, no need to make it a primitive.

start(a::Node)=start(a.value)
done(a::Node,i)=done(a.value,i)
next{A<:AbstractArray}(a::Node{A},i)=(a[i],i+1)
next{T<:Tuple}(a::Node{T},i)=(a[i],i+1)
next{T<:Number}(a::Node{T},i)=(a,true)

# Julia handles access to AbstractArray, Associative, and Tuple
# subtypes using getindex:
#
# getindex(obj, key...) => value
# setindex!(obj, val, key...) => obj
#
# So we can handle these container types by overloading getindex,
# defining getindex and ungetindex (used for gradient) as primitives.

import Base: getindex
@primitive getindex{T<:AbstractArray}(x::Node{T},i...)
@primitive getindex{T<:Associative}(x::Node{T},i...)
@primitive getindex{T<:Tuple}(x::Node{T},i)
getindex(::D1,y,x,i...) = dy->ungetindex(x,dy,i...) # y=x[i], dy=df/dy
getindex{N}(::Dn{N},a...) = 0                 # only the first argument has a gradient

# If y=getindex(x,i...) and we receive dy, we need to create dx as
# with zeros similar to x, with only dx[i] set to dy.  This is what
# ungetindex is for.

ungetindex{T}(x::AbstractArray{T}, dy, i...)=(dx = isbits(T) ? zeros(x) : fill!(Array(Any, size(x)), nothing); setindex!(dx,dy,i...); dx)
ungetindex(x::Tuple, dy, i)            = ntuple(j->(j==i ? dy : nothing), length(x))
ungetindex(x::Associative, dy, i...)   = (dx=similar(x);setindex!(dx,dy,i...);dx)

# In case of a higher order gradient, ungetindex would be called with
# Node inputs and needs to be recorded.
@primitive ungetindex{T<:AbstractArray}(x::Node{T},dy,i...)
@primitive ungetindex{T<:Associative}(x::Node{T},dy,i...)
@primitive ungetindex{T<:Tuple}(x::Node{T},dy,i)

# dx=ungetindex(x,dy,i...) returned dx as all zeros with dx[i]=dy.
# Now we receive the gradient wrt its output: ddx.  To get the
# gradient wrt its dy input, we just need to extract ddx[i].  It is
# not differentiable wrt other inputs.
ungetindex(::D2, dx, x, dy, i...) = ddx->getindex(ddx,i...)
ungetindex{N}(::Dn{N}, dx...) = 0 # only arg 2 has a gradient
