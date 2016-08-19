matmul2arg = Dict{Symbol,Any}(
:* => (:(dy->A_mul_Bc(dy,x2)), :(dy->Ac_mul_B(x1,dy))),
:Ac_mul_B  => (:(dy->A_mul_Bc(x2,dy)), :(dy->x1*dy)),
:A_mul_Bc  => (:(dy->dy*x2), :(dy->Ac_mul_B(dy,x1))),
:Ac_mul_Bc => (:(dy->Ac_mul_Bc(x2,dy)), :(dy->Ac_mul_Bc(dy,x1))),
:At_mul_B  => (:(dy->A_mul_Bt(x2,dy)), :(dy->x1*dy)),
:A_mul_Bt  => (:(dy->dy*x2), :(dy->At_mul_B(dy,x1))),
:At_mul_Bt => (:(dy->At_mul_Bt(x2,dy)), :(dy->At_mul_Bt(dy,x1))),
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

for (_f,_d) in matmul2arg
    testargs(::Fn{_f},t1,t2)=(rand(2,2),rand(2,2))
end
