# broadcast2arg:
# These functions use broadcasting to handle arrays of different sizes.
# Unless otherwise specified they support:
# (N,N) (N,A) (A,N) (A,A) (A,B)
# where N:Number, A,B arrays of broadcast compatible sizes.

broadcast2arg = Dict{Symbol,Any}(
:.+ => (:dy,:dy),                    # extra (A,)
:.* => (:(dy.*x2),:(dy.*x1)),                # extra (A,)
:.- => (:dy,:(-dy)),
#:.% => (:dy,:(dy.*(-trunc(x1./x2)))),  # BUG: WARNING: (:check_grads,(:sum,:.%),:args,([-1.6685861285973334,2.349598738753782],[0.5880954718832765,-0.0010728600840855926]),:exact,([1.0,1.0],[2.0,2190.0]),:numeric,([1.0000000000021103,-9.728600840858691],[1.9999999999997797,-4.863172375468294])), WARNING: (:check_grads,(:sum,:.%),:args,([0.20579984208295538,-0.5521335915808314],[0.14504947039368943,-5.795215813098871e-5]),:exact,([1.0,1.0],[-1.0,-9527.0]),:numeric,([0.9999999999998899,-0.15904316261985962],[-0.9999999999998899,0.5895451080050601]))
:./ => (:(dy./x2),:(-dy.*x1./abs2(x2))),
:.\ => (:(-dy.*x2./abs2(x1)),:(dy./x1)),
:.^ => (:((x2==0 ? dy.*0 : x2==1 ? dy : x2==2 ? dy.*2.*x1 : dy.*x2.*x1.^(x2-1))),
        :(dy.*y.*log(x1))), # domain: x1 >= 0 (unless we use complex args)
#:.<< => :todo,                   # domain: Integers, left bit shift; operators,arraymath,broadcast
#:.>> => :todo,                   # domain: Integers, right bit shift
)

for (f,g) in broadcast2arg
    @eval @primitive $f(x1,x2),dy,y  unbroadcast(x1,$(g[1]))  unbroadcast(x2,$(g[2]))
end

broadcast2cmp = [
:.<,
:.<=,
:.==,
:.>,
:.>=,
]                 

for f in broadcast2cmp
    @eval begin
        # To avoid conflict at broadcast.jl:414
        $f(x1::AbstractArray,x2::Value)=$f(x1,x2.value)
        $f(x1::Value,x2::AbstractArray)=$f(x1,x2.value)
        @zerograd $f(x1,x2)
    end
end


fixdomain(::Fn{:.^},x1,x2)=(abs(x1),x2)

# Other functions in broadcast.jl:

# eval
# droparg1: Not exported
# longer_tuple: Not exported
# longer_size: Not exported
# broadcast_shape: Not exported
# check_broadcast_shape: Not exported
# gen_broadcast_body_cartesian: Not exported
# gen_broadcast_body_iter: Not exported
# bpack: Not exported
# dumpbitcache: Not exported
# gen_broadcast_body_cartesian_tobitarray: Not exported
# gen_broadcast_body_iter_tobitarray: Not exported
# gen_broadcast_function: Not exported
# gen_broadcast_function_tobitarray: Not exported
# broadcast!
# broadcast
# bitbroadcast
# broadcast!_function
# broadcast_function
# broadcast_getindex
# broadcast_getindex!: Not exported
# broadcast_setindex!
# .*
# .%
# .<<
# .>>
# eltype_plus: Not exported
# .+
# type_minus: Not exported
# .-
# type_div: Not exported
# ./
# .\
# type_rdiv: Not exported
# .//
# type_pow: Not exported
# .^
# $(Expr(:$, :f)): Not a symbol
# $(Expr(:$, :bitf)): Not a symbol
# $(Expr(:$, :cachef)): Not a symbol
# bitcache_pow: Not exported
