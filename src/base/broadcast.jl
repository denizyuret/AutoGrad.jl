# broadcast2arg:
# These functions use broadcasting to handle arrays of different sizes.
# Unless otherwise specified they support:
# (N,N) (N,A) (A,N) (A,A) (A,B)
# where N:Number, A,B arrays of broadcast compatible sizes.

broadcast2arg = Dict{Symbol,Any}(
:.+ => (1,1),                    # extra (A,)
:.* => (:x2,:x1),                # extra (A,)
:.- => (1,-1),
:.% => (1,:(-trunc(x1./x2))),
:./ => (:(1./x2),:(-x1./abs2(x2))),
:.\ => (:(-x2./abs2(x1)),:(1./x1)),
:.^ => (:(x2.*x1.^(x2-1)),:(y.*log(x1))), # domain: x1 >= 0 (unless we use complex args)
)

for (_f,_d) in broadcast2arg
    @eval begin
        @primitive $_f(x1::Node, x2::Node)
        @primitive $_f(x1::Node, x2::Union{Number,AbstractArray})
        @primitive $_f(x1::Union{Number,AbstractArray},x2::Node)
        $_f(::D1, y, x1, x2)=unbroadcast(y, x1, (dy->dy.*$(_d[1])))
        $_f(::D2, y, x1, x2)=unbroadcast(y, x2, (dy->dy.*$(_d[2])))
    end
end

broadcast2zerograd = Dict{Symbol,Any}(
:.<< => :todo,                   # domain: Integers, left bit shift; operators,arraymath,broadcast
:.>> => :todo,                   # domain: Integers, right bit shift
:.<  => 0,
:.<= => 0,
:.== => 0,
:.>  => 0,
:.>= => 0,
)
