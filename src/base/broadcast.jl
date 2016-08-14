# broadcast2arg:
# These functions use broadcasting to handle arrays of different sizes.
# Unless otherwise specified they support:
# (N,N) (N,A) (A,N) (A,A) (A,B)
# where N:Number, A,B arrays of broadcast compatible sizes.

broadcast2arg = Dict{Symbol,Any}(
:.+ => (1,1),                    # extra (A,), BUG: MethodError(isapprox,([1.0,1.0],2.0000000000042206))
:.* => (:x2,:x1),                # extra (A,), BUG: MethodError(isapprox,([0.3756898299727047,-0.6644013315217091],-0.28871150154874403))
:.- => (1,-1), # BUG: MethodError(isapprox,([-1.0,-1.0],-2.0000000000042206))
:.% => (1,:(-trunc(x1./x2))),  # BUG: MethodError(isapprox,([-1.0,-1.0],-1.9999999999997797))
:./ => (:(1./x2),:(-x1./abs2(x2))), # BUG: MethodError(isapprox,([0.551017029971155,0.733939349058221],1.284956382050506))
:.\ => (:(-x2./abs2(x1)),:(1./x1)), # BUG: MethodError(isapprox,([-0.45498059782573336,-1.0331589803808776],-1.48813957820626))
:.^ => (:(x2.*x1.^(x2-1)),:(y.*log(x1))), # domain: x1 >= 0 (unless we use complex args)
# :.<  => 0, # BUG: MethodError(isless,(0.3836010032871748,N31(0.28940741417311505,(:A3,:R61))))
# :.<= => 0, # BUG: MethodError(isless,(N64(0.2943038720867562,(:A93,:R0)),1.6879170322331594))
# :.== => 0, # BUG: StackOverflowError()
# :.>  => 0, # BUG: StackOverflowError()
# :.>= => 0, # BUG: StackOverflowError()
#:.<< => :todo,                   # domain: Integers, left bit shift; operators,arraymath,broadcast
#:.>> => :todo,                   # domain: Integers, right bit shift
)

defgrads(broadcast2arg, Number, Number)
defgrads(broadcast2arg, AbstractArray, Number)
defgrads(broadcast2arg, Number, AbstractArray)
defgrads(broadcast2arg, AbstractArray, AbstractArray)

testargs(::Type{Val{:.^}},x...)=map(abs,testargs(nothing,x...))
