matmul2arg = Dict{Symbol,Any}(
:* => (:(A_mul_Bc(dy,x2)), :(Ac_mul_B(x1,dy))),
:Ac_mul_B  => (:(A_mul_Bc(x2,dy)), :(x1*dy)),
:A_mul_Bc  => (:(dy*x2), :(Ac_mul_B(dy,x1))),
:Ac_mul_Bc => (:(Ac_mul_Bc(x2,dy)), :(Ac_mul_Bc(dy,x1))),
:At_mul_B  => (:(A_mul_Bt(x2,dy)), :(x1*dy)),
:A_mul_Bt  => (:(dy*x2), :(At_mul_B(dy,x1))),
:At_mul_Bt => (:(At_mul_Bt(x2,dy)), :(At_mul_Bt(dy,x1))),
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

# defgrads(matmul2arg, AbstractVecOrMat, AbstractVecOrMat; dymul=false)

for (f,d) in matmul2arg
    @eval @primitive $f(x1,x2),dy,y $(d[1]) $(d[2])
    fixdomain(::Fn{f},t1,t2)=(rand(2,2),rand(2,2))
end


# TODO:

# eval
# arithtype: Not exported
# scale!
# scale
# vecdot
# dot
# Ac_mul_B
# At_mul_B
# *
# A_mul_B!
# At_mul_B!
# Ac_mul_B!
# A_mul_Bt
# A_mul_Bt!
# At_mul_Bt
# At_mul_Bt!
# A_mul_Bc
# A_mul_Bc!
# Ac_mul_Bc
# Ac_mul_Bc!
# Ac_mul_Bt: Not exported
# Ac_mul_Bt!: Not exported
# copytri!: Not exported
# gemv!: Not exported
# syrk_wrapper!: Not exported
# herk_wrapper!: Not exported
# gemm_wrapper: Not exported
# gemm_wrapper!: Not exported
# lapack_size: Not exported
# copy!
# copy_transpose!: Not exported
# generic_matvecmul!: Not exported
# generic_matmatmul: Not exported
# generic_matmatmul!: Not exported
# _generic_matmatmul!: Not exported
# matmul2x2: Not exported
# matmul2x2!: Not exported
# matmul3x3: Not exported
# matmul3x3!: Not exported
