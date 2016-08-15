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

function testargs(::Fn{:*}, a...)
    x = map(a) do ai
        ai <: AbstractVector ? rand(2) :
        ai <: AbstractMatrix ? (a[1] <: AbstractVector ? rand(1,2) : rand(2,2)) :
        nothing
    end
    if in(nothing, x)
        return testargs(Fn2(:*), a...)
    else
        return x
    end
end
