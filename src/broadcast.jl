# This should stop broadcast from fusing Recs:

# Alternatives:
# - define your own bcast function.
# - keep a dictionary of zerograd functions.
# - need some way to define gradients, and some way to define zero gradients

import Base.Broadcast: broadcasted
broadcast_r = recorder(broadcasted)
broadcasted(f, x::Rec) = broadcast_r(f,x)
broadcasted(f, x::Rec, y...) = broadcast_r(f,x,y...) # useful for clamp
broadcasted(f, x::Rec, y) = broadcast_r(f,x,y)
broadcasted(f, x, y::Rec) = broadcast_r(f,x,y)
broadcasted(f, x::Rec, y::Rec) = broadcast_r(f,x,y)

# This fixes sum(x.*x,dims=1) giving MethodError: no method matching sum(::Base.Broadcast.Broadcasted; dims=1)
import Base.Broadcast: materialize
materialize_r = recorder(materialize)
materialize(x::Rec) = materialize_r(x)
materialize(::Type{Grad{1}},dy,y,x::Rec) = dy

# The way broadcasting works in Julia:
# y = f(x...) where f is a broadcasting operation.
# size(y) = broadcast_shape(x...)
# ndims(y) = max ndims(x)
# size(y,i) = max size(x,i)
# size(x,i) = 1 or size(y,i) for all x and i<=ndims(x)
# if ndims(x) < ndims(y) the extra dimensions of x are treated as 1

"""
    unbroadcast(x,dx)

Bring dx to x's size via unbroadcasting (reduction). This is needed
when defining gradients of multi-argument broadcasting functions where
the arguments and the result may be of different sizes.

"""
function unbroadcast(x, dx)
    if size(x)==size(dx)
        return dx
    elseif isa(getval(x),Number)
        return sum(dx)
    elseif isa(getval(dx),Number)
        return fill!(similar(getval(x)),dx)
    else
        d = []
        for i=1:ndims(dx)
            size(x,i) == size(dx,i) > 1 && continue
            size(x,i) != 1 && throw(DimensionMismatch())
            push!(d,i)
        end
        length(d)==1 && (d=d[1])
        return reshape(sum(dx, dims=d), size(x))
    end
end

# This fixes unbroadcasted when x isa Broadcasted
using Base.Broadcast: Broadcasted
import Base: size
size(x::Broadcasted,i...)=length.(axes(x,i...))
