#### The way broadcasting works in Julia (from base/broadcast.jl):
# 
### Dot notation is lazy, explicit broadcast is not:
# (a .+ b .* c) => materialize(broadcasted(+, a, broadcasted(*, b, c)))
# sin.(cos.(x)) => materialize(broadcasted(sin, broadcasted(cos, x)))
# broadcast(sin, broadcast(cos, x)) => materialize(broadcasted(sin, materialize(broadcasted(cos, x))))
# 
### broadcasted creates a Broadcasted structure unless overriden:
# broadcasted(f,x...) => (xx = broadcastable.(x); broadcasted(combine_styles(xx...),f,xx...))
# broadcasted(::{S<:BroadcastStyle}, f, args...) = Broadcasted{S}(f, args)
#
### Broadcasted is a 4 parameters struct with 3 members:
# Broadcasted{Style}(f::F, args::Args, axes::Axes=nothing) =
# Broadcasted{Style, Axes, F, Args}(f, args, axes)
#
### materialize calculates the actual result using copy:
# materialize(bc::Broadcasted) = copy(instantiate(bc))
# materialize(x) = x
#
### instantiate: adds or checks the Axes component:
# instantiate(bc::Broadcasted{S}) = Broadcasted{S}(bc.f, bc.args, combine_axes(bc.args...))
# instantiate(x) = x
# 
### copy: allocates result container and fills it using copyto!
# copy(bc::Broadcasted) = copyto!(similar(bc, etype), bc)
# similar(bc::Broadcasted, ::Type{T}) = similar(Array{T}, axes(bc))
#
### copyto!: is responsible for calculating the final result

# Current design avoids generating Broadcasted objects:

# For regular primitives:
# f(x...) => forw(f,x...)           # (macros.jl; @primitive f if any x is a Value)
# forw(f,x...) => f(value.(x)...)   # (core.jl; f is the recorded function)
# back(f,i,dy,y,x...)               # (macros.jl; @primitive defines this)

# For broadcasted primitives:
# f.(x...) => broadcasted(f,x...)                       # (parser)
# broadcasted(f,x...) => forw(broadcast,f,x...)         # (macros.jl; @primitive defines this, @primitive1 does not)
# forw(broadcast,f,x...) => broadcast(f,value.(x)...)   # (core.jl; broadcast is the recorded function)
# back(broadcast,i,dy,y,f,x...)                         # (macros.jl; @primitive defines this, @primitive1 does not)

# For direct use of broadcast:
# broadcast(f,x...) => materialize(broadcasted(f,x...)) # (base/broadcast.jl fallback for unknown Value types)
# broadcasted(f,x...) => forw(broadcast,f,x...)         # (forw calling broadcast with unboxed args returning Value)
#DEPRECATED: materialize(x::Value) => x                            # (broadcast.jl; defined below)

import .Broadcast: broadcasted, materialize, broadcastable, BroadcastStyle, Style

# To catch whenever one arg is a Value in broadcast expressions, we define a style:
BroadcastStyle(::Type{<:Value}) = Style{Value}()
BroadcastStyle(s::Style{Value}, ::BroadcastStyle) = s
broadcastable(x::Value) = x     # This is necessary, default is collect(x) which loses Value

# For user-defined functions we want to call the function with marked arguments:
struct Bcasted; value; end
bval(x::Bcasted) = x.value
bval(x) = x

function broadcasted(::Style{Value}, f, args...)
    if isempty(_tapes)
        broadcasted(f, value.(args)...)
    else
        bval(f(Bcasted.(args)...))
    end
end

# For each primitive function this needs to be overwritten:
# (1) we want to call forw:
#    broadcasted(::Style{Value}, f::typeof(sin), x) = forw(broadcast, f, x)
# (2) we want to handle Bcasted args:
#    sin(a::Bcasted) = Bcasted(sin.(bval(a)))

# Deprecated:
# broadcasted(f, x::Value, y...) = forw(broadcast,f,x,y...)
# broadcasted(f, x, y::Value, z...) = forw(broadcast,f,x,y,z...) # useful for x.^2 => broadcasted(literal_pow,^,x,Val(2))
# broadcasted(f, x::Value, y::Value, z...) = forw(broadcast,f,x,y,z...) # ambiguity fix
# materialize(x::Value)=x  # This fixes sum(x.*x,dims=1) giving MethodError: no method matching sum(::Base.Broadcast.Broadcasted; dims=1)

# Design alternatives:
# - define your own bcast function.
# - keep a dictionary of zerograd functions.
# - need some way to define gradients, and some way to define zero gradients

# Broadcasting dimensions:
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
    elseif isa(value(x),Number)
        return sum(dx)
    elseif isa(value(dx),Number)
        return fill!(similar(value(x)),dx)
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
