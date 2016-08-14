matmul2arg = Dict{Symbol,Any}(
:* => (:x2,:x1),                   # (N,) (M,) (N,*) (*,N) (M,V) (M,M)
)

# Methods for multiplication:
# *(x::Float64, y::Float64) at float.jl:212  (same as .*)
# *(A::Number, B::AbstractArray{T,N}) at abstractarraymath.jl:54  (calls .*)
# *(A::AbstractArray{T,N}, B::Number) at abstractarraymath.jl:55  (calls .*)
# *(A::AbstractArray{T,2}, B::AbstractArray{T,1}) at linalg/matmul.jl:82
# *(A::AbstractArray{T,1}, B::AbstractArray{T,2}) at linalg/matmul.jl:89  (works only if size(B,1)==1)
# *(A::AbstractArray{T,2}, B::AbstractArray{T,2}) at linalg/matmul.jl:131

@primitive *(x1::Node, x2::Node)
@primitive *(x1::Node, x2::Union{Number,AbstractArray})
@primitive *(x1::Union{Number,AbstractArray},x2::Node)

# Gradients:
# For the first three cases the gradient is identical to .*
# i.e. grad1=dy.*x2, grad2=dy.*x1

*{A<:Number,B<:Number,C<:Number}(::D1, y::Node{A}, x1::Nval{B}, x2::Nval{C})=(dy->dy*x2)
*{A<:Number,B<:Number,C<:Number}(::D2, y::Node{A}, x1::Nval{B}, x2::Nval{C})=(dy->dy*x1)

*{A<:AbstractArray,B<:Number}(::D1, y::Node{A}, x1::Nval{B}, x2::Nval{A})=unbroadcast(y, x1, (dy->dy.*x2))
*{A<:AbstractArray,B<:Number}(::D1, y::Node{A}, x1::Nval{A}, x2::Nval{B})=unbroadcast(y, x1, (dy->dy.*x2))
*{A<:AbstractArray,B<:Number}(::D2, y::Node{A}, x1::Nval{B}, x2::Nval{A})=unbroadcast(y, x2, (dy->dy.*x1))
*{A<:AbstractArray,B<:Number}(::D2, y::Node{A}, x1::Nval{A}, x2::Nval{B})=unbroadcast(y, x2, (dy->dy.*x1))

# For the last three cases we have matrix multiplication:
# grad1=dy*x2' grad2=x1'*dy

*{A<:AbstractArray,B<:AbstractArray,C<:AbstractArray}(::D1, y::Node{A}, x1::Nval{B}, x2::Nval{C})=(dy->dy*x2')
*{A<:AbstractArray,B<:AbstractArray,C<:AbstractArray}(::D2, y::Node{A}, x1::Nval{B}, x2::Nval{C})=(dy->x1'*dy)
