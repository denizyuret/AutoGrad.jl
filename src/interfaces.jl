# Here we will define iteration (start,done,next) and indexing
# (getindex,setindex!,endof) interfaces for generic Node types.

# Specific types like AbstractArray and Associative need only define
# the next, endof, and ungetindex methods.

# Iteration is used in `for x in a` loops and for `(x,y)=a` multiple
# assignments.

# start(a) => initial state
# done(a,state) => whether state s is final
# next(a,state) => (element, nextState)

# start and done return state and bool, not differentiable.
# next returns a tuple with an element and needs to be defined.
start(a::Node)=start(a.value)
done(a::Node,i)=done(a.value,i)
next(a::Node,i)=throw(MethodError(next,(a,i)))

# Julia handles access to AbstractArray, Associative, and Tuple
# subtypes using getindex:

# getindex(obj, key...) => value
# setindex!(obj, val, key...) => obj

# So we can handle these container types by overloading getindex,
# defining getindex and ungetindex (used for gradient) as primitives.

@primitive getindex(x::Node,i...)
getindex(::D1,y::Node,x::Node,i...) = dy->ungetindex(x,dy,i...) # y=x[i], dy=dJ/dy

# setindex! is an overwriting operation and not supported for now:
setindex!(x::Node,i...)=error("Overwriting operations currently not supported.")

# endof is type specific:
endof(x::Node)=throw(MethodError(endof,(x,)))

# If y=getindex(x,i...) and we receive dy, we need to create dx as
# with zeros similar to x, with only dx[i] set to dy.  The user should
# define ungetindex for specific types.

ungetindex(x,dy,i...)=throw(MethodError(ungetindex,x))

# In case of a higher order gradient, ungetindex would be called with
# Node inputs and needs to be recording primitive.

@primitive ungetindex(x::Node,dy,i...)

# dx=ungetindex(x,dy,i...) returned dx as all zeros with dx[i]=dy.
# Now we receive the gradient wrt its output: ddx.  To get the
# gradient wrt its dy input, we just need to extract ddx[i].  It is
# not differentiable wrt other inputs.

ungetindex(::D2,dx,x,dy,i...) = ddx->getindex(ddx,i...)


### DEAD CODE:

# Here are some examples for next, actual definitions are under base.
# next can be defined in terms of getindex, no need to make it a primitive.
# next{A<:AbstractArray}(a::Node{A},i)=(a[i],i+1)
# next{T<:Tuple}(a::Node{T},i)=(a[i],i+1)
# next{T<:Number}(a::Node{T},i)=(a,true)

# We no longer define getindex for each type as a primitive:
# import Base: getindex
# @primitive getindex{T<:Associative}(x::Node{T},i...)
# @primitive getindex{T<:Tuple}(x::Node{T},i)
# getindex(::D1,y,x,i...) = dy->ungetindex(x,dy,i...) 
# getindex{N}(::Dn{N},a...) = 0                 # only the first argument has a gradient

# ungetindex exampes:
# ungetindex(x::Tuple, dy, i)            = ntuple(j->(j==i ? dy : nothing), length(x))
# ungetindex(x::Associative, dy, i...)   = (dx=similar(x);setindex!(dx,dy,i...);dx)

# In case of a higher order gradient, ungetindex would be called with
# Node inputs and needs to be recorded.
# @primitive ungetindex{T<:Associative}(x::Node{T},dy,i...)
# @primitive ungetindex{T<:Tuple}(x::Node{T},dy,i)

# ungetindex{N}(::Dn{N}, dx...) = 0 # only arg 2 has a gradient
