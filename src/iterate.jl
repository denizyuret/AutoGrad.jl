import Base: iterate, indexed_iterate

### ITERATION: We define iterate in terms of getindex which does the necessary tracking.

# Iterate is used in `for x in a` loops 
iterate(a::Value,i=nothing)=throw(MethodError(iterate,(a,i)))
iterate(a::Value{T},i=1) where {T<:AbstractArray} = i > length(a) ? nothing : (a[i],i+1)
iterate(a::Value{T},i=1) where {T<:Tuple}         = i > length(a) ? nothing : (a[i],i+1)
iterate(a::Value{T})   where {T<:AbstractDict} = (v=iterate(value(a));   v==nothing ? v : (((k,_),j)=v;(k=>a[k],j)))
iterate(a::Value{T},i) where {T<:AbstractDict} = (v=iterate(value(a),i); v==nothing ? v : (((k,_),j)=v;(k=>a[k],j)))
iterate(a::Value{T})   where {T<:Number} = (a,nothing)
iterate(a::Value{T},i) where {T<:Number} = nothing

# indexed_iterate for `(x,y)=a` multiple assignments.
indexed_iterate(a::Value{T},i::Int,state=1) where {T<:Tuple} = (a[i],i+1)
indexed_iterate(a::Value{T},i::Int,state=1) where {T<:Array} = (a[i],i+1)

