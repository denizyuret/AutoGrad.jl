import Base: iterate, indexed_iterate

### ITERATION: We define iterate in terms of getindex which does the necessary tracking.

# Iterate is used in `for x in a` loops 
iterate(a::Rec,i=nothing)=throw(MethodError(iterate,(a,i)))
iterate(a::Rec{T},i=1) where {T<:AbstractArray} = i > length(a) ? nothing : (a[i],i+1)
iterate(a::Rec{T},i=1) where {T<:Tuple}         = i > length(a) ? nothing : (a[i],i+1)
iterate(a::Rec{T})   where {T<:AbstractDict} = (v=iterate(a.value);   v==nothing ? v : (((k,_),j)=v;(k=>a[k],j)))
iterate(a::Rec{T},i) where {T<:AbstractDict} = (v=iterate(a.value,i); v==nothing ? v : (((k,_),j)=v;(k=>a[k],j)))
iterate(a::Rec{T})   where {T<:Number} = (a,nothing)
iterate(a::Rec{T},i) where {T<:Number} = nothing

# indexed_iterate for `(x,y)=a` multiple assignments.
indexed_iterate(a::Rec{T},i::Int,state=1) where {T<:Tuple} = (a[i],i+1)
indexed_iterate(a::Rec{T},i::Int,state=1) where {T<:Array} = (a[i],i+1)

