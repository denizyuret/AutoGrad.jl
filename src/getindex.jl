import Base: getindex, setindex!, get, view, selectdim

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

# We handle the containers by overloading getindex:

@primitive  getindex(x,i...),dxi,xi  ungetindex(x,dxi,i)
back(::typeof(getindex),::Type{Arg{N}},o...) where {N} = nothing # Only the first arg has gradient

# use ungetindex machinery also for view and selectdim
@primitive  view(x,i...),dxi,xi  ungetindex(x,dxi,i)
back(::typeof(view),::Type{Arg{N}},o...) where {N} = nothing # Only the first arg has gradient
@inline selectdim(A::Value{<:AbstractArray}, d::Integer, i) = Base._selectdim(A, d, i, Base.setindex(map(Base.Slice, axes(A)), i, d))

# Gradient of getindex: If xi=getindex(x,i...) and we receive dxi,
# ungetindex creates dx representing zeros similar to x, with only
# dx[i...] set to dxi.  We use the sparse container Sparse for
# efficiency when x is an array.
# x -> getindex -> xi -> grad -> dxi -> ungetindex -> dx -> grad -> ddx -> getindex -> ddxi

# ungetindex(x,dxi,i)=Sparse(x,[dxi],[i])

# For Object arrays, Dict, Tuple, Number no need to use Sparse:

ungetindex(x::Number,dxi,i)=dxi

ungetindex(x::AbstractDict,dxi,i)=setindex!(empty(x), dxi, i...)

function ungetindex(x::Tuple,dxi,i) # use array code in case there are repeated indices
    dx = Array{Any}(nothing, length(x))
    dxi = (i[1] isa Real ? dxi : collect(Any,dxi))
    addtoindex!(dx, dxi, Base.to_indices(dx,i)...)
    tuple(dx...)
end

function ungetindex(x::AbstractArray{T},dxi,i) where T
    if isbitstype(T)
        # Issue Knet#439: hessians for neural networks In higher order derivatives, Sparse
        # structs may participate in operations like matmul etc.  We do not want to
        # implement every possible operation with Sparse, so in these cases we generate the
        # full array instead.  The first two conditions should only trigger for higher order
        # gradients, not during regular training.
        if dxi isa Value
            forw(addto!, zero(x), forw(ungetindex, x, dxi, i))
        elseif recording()
            addtoindex!(zero(x), dxi, i...)
        else
            Sparse(x,[dxi],[i])
        end
    else
        # Using addtoindex! instead of setindex! to handle repeated indices
        addtoindex!(Array{Union{T,Nothing}}(nothing, size(x)), dxi, i...)
    end
end

# For higher order derivatives, the operation of ungetindex might be
# recorded and differentiated, so it must be a primitive.  It is only
# differentiable wrt its value arg.  It should unbox its arguments,
# but it only needs to record if the value argument is boxed.  We'll
# have to define this manually.  To unbox the container arg and
# resolve ambiguity the ungetindex methods cover all combinations of
# first two args:
# (a,a), (a,r), (r,r), (r,a), (g2,a), (g2,r), (g,a), (g,r)

# Ignore Value in the first and third position
ungetindex(x::Value,dxi::Value,i)=ungetindex(value(x),dxi,value(i))
ungetindex(x::Value,dxi,i)=ungetindex(value(x),dxi,value(i))

back(::typeof(ungetindex),::Type{Arg{2}},ddx,dx,x,dxi,i)=getindex(ddx,value(i)...)
back(::typeof(ungetindex),::Type{Arg{N}},o...) where {N} = nothing

# gradcheck works with the first arg, we need to check ungetindex grad for its second arg
# ungetindex2(value, container, index)=ungetindex(container, value, index)
# addtest(:ungetindex2, rand(),  rand(2),   (2,))
# addtest(:ungetindex2, rand(2), rand(3),   (2:3,))
# addtest(:ungetindex2, rand(),  rand(2,2), (1,2))
# addtest(:ungetindex2, rand(2), rand(3,3), (1:2,3))

# get (getindex with a default value)
# This can be left as a composite function, it will get its gradient from getindex if necessary.
get(A::Value{T}, i::Integer, default) where {T<:AbstractArray} = (if checkbounds(Bool, length(A), i); A[i]; else; default; end)
get(A::Value{T}, I::Tuple{}, default) where {T<:AbstractArray} = similar(A, typeof(default), 0)
get(A::Value{T}, I::Dims, default) where {T<:AbstractArray}    = (if checkbounds(Bool, size(A), I...); A[I...]; else; default; end)
