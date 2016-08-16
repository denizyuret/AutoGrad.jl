matmul2arg = Dict{Symbol,Any}(
:* => (:(dy->dy*x2'), :(dy->x1'*dy)),
)

# Methods for multiplication:
# *(x::Float64, y::Float64) at float.jl:212  (same as .*)
# *(A::Number, B::AbstractArray{T,N}) at abstractarraymath.jl:54  (calls .*)
# *(A::AbstractArray{T,N}, B::Number) at abstractarraymath.jl:55  (calls .*)
# *(A::AbstractArray{T,2}, B::AbstractArray{T,1}) at linalg/matmul.jl:82
# *(A::AbstractArray{T,1}, B::AbstractArray{T,2}) at linalg/matmul.jl:89  (works only if size(B,1)==1)
# *(A::AbstractArray{T,2}, B::AbstractArray{T,2}) at linalg/matmul.jl:131
#
# The first three are handled by base/float.
# The final three implement matrix multiplication.
# We need to handle these manually instead of calling defgrads because of the different gradient form.
# defgrads(matmul2arg, AbstractVecOrMat, AbstractVecOrMat)
# grad1=dy*x2' grad2=x1'*dy

defgrads(matmul2arg, AbstractVecOrMat, AbstractVecOrMat; dymul=false)

function testargs{T1<:AbstractVecOrMat,T2<:AbstractVecOrMat}(::Fn{:*},t1::Type{T1},t2::Type{T2})
    x1 = (t1 <: AbstractVecOrMat ? (randn() < 0.5 ? randn(2) : randn(2,2)) :
          t1 <: AbstractMatrix ? randn(2,2) :
          t1 <: AbstractVector ? randn(2) :
          error("testargs(*,$t1,$t2)"))
    x2 = (ndims(x1)==1 ? rand(1,2) : 
          t2 <: AbstractVecOrMat ? (randn() < 0.5 ? randn(2) : randn(2,2)) :
          t2 <: AbstractMatrix ? randn(2,2) :
          t2 <: AbstractVector ? randn(2) : 
          error("testargs(*,$t1,$t2)"))
    return (x1,x2)
end

