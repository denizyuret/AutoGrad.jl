_Ac_mul_B(A,B) = adjoint(A) * B
_At_mul_B(A,B) = transpose(A) * B
_A_mul_Bc(A,B) = A * adjoint(B)
_A_mul_Bt(A,B) = A * transpose(B)
_Ac_mul_Bc(A,B) = adjoint(A) * adjoint(B)
_At_mul_Bt(A,B) = transpose(A) * transpose(B)

matmul2arg = [
(:*,         :(dy*x2'),    :(x1'*dy),    :(dy),   :(dy)),
(:_Ac_mul_B,  :(x2*dy'),    :(x1*dy),     :(dy'),  :(dy)),
(:_At_mul_B,  :(x2*transpose(dy)),   :(x1*dy),     :(transpose(dy)), :(dy)),
(:_A_mul_Bc,  :(dy*x2),     :(dy'*x1),    :(dy),   :(dy')),
(:_A_mul_Bt,  :(dy*x2),     :(transpose(dy)*x1),   :(dy),   :(transpose(dy))),
(:_Ac_mul_Bc, :(x2'*dy'),   :(dy'*x1'),   :(dy'),  :(dy')),
(:_At_mul_Bt, :(transpose(x2)*transpose(dy)), :(transpose(dy)*transpose(x1)), :(transpose(dy)), :(transpose(dy))),
]

function addtest_matmul(f)
    addtestN(f,rand(2,2),rand(2,2))
    addtestN(f,rand(2,2),rand())
    addtestN(f,rand(),rand(2,2))
    addtestN(f,rand(),rand())
end

# We need to handle scalars in Ac_mul_B etc. as well
# To distinguish these from the Array-Array case, we type the scalars with Number
# We leave array cases untyped to allow for extensions like KnetArray
# Using sum to handle scalars and reshape to handle vectors
for (f,g1,g2,dy1,dy2) in matmul2arg
    @eval @primitive $f(x1::Number,x2::Number),dy,y  dy*x2                       dy*x1
    @eval @primitive $f(x1::Number,x2),dy,y  	     sum($dy2.*x2)               reshape($dy2.*x1,size(x2))
    @eval @primitive $f(x1,x2::Number),dy,y          reshape($dy1.*x2,size(x1))  sum($dy1.*x1)
    @eval @primitive $f(x1,x2),dy,y                  reshape($g1,size(x1))       reshape($g2,size(x2))
    addtest_matmul(f)
end


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


# TODO:

# eval
# arithtype: Not exported
# scale!
# scale
# vecdot
@primitive dot(x1, x2),dy,y  dy*x2  dy*x1
addtestN(:dot, rand(3), rand(3))

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
