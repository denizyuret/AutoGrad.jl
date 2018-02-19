# Julia6 broadcast fix based on:
# [1] https://github.com/MikeInnes/TakingBroadcastSeriously.jl (@MikeInnes)
# [2] https://github.com/JuliaLang/julia/issues/22060#issuecomment-304294397 (@ylxdzsw)

# 1. In Julia6+ broadcasting functions are called with a dot:
# > sin(x::Number) but sin.(x::Array)
# The dotted function call is automatically converted to a broadcast call:
# > f.(x) => broadcast(f, x)

# 2. Furthermore multiple dotted functions are automatically fused:
# > sin.(cos.(x)) => broadcast(F, x) where F is (sin o cos) to be applied to each element.
# Containers like KnetArray or Rec will not know how to handle composite functions like F.
# One solution is to define a generic broadcast fallback for our type:
# > broadcast(F, x::T) => F(Broadcasted(x)).value
# We need to do this for arbitrary number of args, the more general solution:
# > broadcast(F, x::Union{previous_types,T}...) = F(Broadcasted.(x)...).value

# 3. F(Broadcasted(x)).value will get turned into:
# > sin(cos(Broadcasted(x))).value
# So each primitive function needs to handle Broadcasted:
# > f(x::Broadcasted) = Broadcasted(bf(x.value)) where bf is the broadcasting version of f.

# In summary, for every new type T and for every primitive f we need to define:
# 1. broadcast(F, x::Union{previous_types,T}...) = F(Broadcasted.(x)...).value
# 2. bf(x::T) needs to be implemented (after being imported)

type Broadcasted{T}
    value::T
end
getval(x::Broadcasted)=x.value
# We need this to not override regular broadcast(f, A, Bs...):
using Base.Broadcast: broadcast_c, containertype
broadcast(f, x::Union{Number,AbstractArray}...)=broadcast_c(f, containertype(x...), x...)
# This captures cases where at least one arg is a Rec:
broadcast(f, x::Union{Number,AbstractArray,Rec}...)=f(Broadcasted.(x)...).value

# broadcast_func(f) gets called with every primitive function in AutoGrad.

function broadcast_func(f)
    f = Symbol(lstrip(string(f), '.'))
    bf = Symbol("broadcast#", f)
    if !isdefined(AutoGrad, bf); @eval begin
        # We need this when x is of a regular type (@primitive only defines bf for Rec)
        $bf(x...) = broadcast($f, x...)
        $f(x::Broadcasted...) = $bf(getval.(x)...) |> Broadcasted
        # We need the following because sometimes the interpreter does not convert all args to Broadcasted:
        $f(x1::Broadcasted, x2) = $bf(getval(x1), x2) |> Broadcasted
        $f(x1, x2::Broadcasted) = $bf(x1, getval(x2)) |> Broadcasted
    end; end
    bf
end


### DEAD CODE:

# sign.(Rec(a))
# => broadcast(sign, Rec(a))
# => sign(Broadcasted(Rec(a))).value
# => broadcast#sign(Rec(a)) |> Broadcasted
# => broadcast#sign(a) # due to @zerograd def
# => broadcast(sign, a)

# Consider:
# f: primitive function
# bf: broadcast_func(f)
# rbf: recorder(bf)
# x: regular array
# rx: Rec(x)
# brx: Broadcasted(rx)

# Here is how it goes down for a primitive function f:
# f.(rx) ## typed by user
# => broadcast(f, rx) ## converted by julia
# => f(brx).value ## util.jl:370
# => bf(rx) |> Broadcasted ## util.jl:383
# => rbf(x) ## broadcast.jl:21 @primitive
# => bf(x)  ## core.jl:123 recorder()
# => broadcast(f, x) ## util.jl:380

# If we have a composite function:
# g.(f.(rx)) ## typed by user, f and g primitives
# => broadcast(h, rx) ## converted by julia, h=x->g(f(x))
# => h(brx).value ## util.jl:370
# => g(f(brx)) ### brx treated like a scalar
# => f(brx) ## util.jl:383 returns a bry
# => g(bry) ## util.jl:383 returns a brz
# => rz ## returned as h(brx).value


#=
            # We need this when x is of a regular type (@primitive only defines bf for Rec)
            $bf(x...) = broadcast($f, x...)
            $f(x::Broadcasted...) = broadcast($f, getval.(x)...) |> Broadcasted
            # We need the following because sometimes the interpreter does not convert all args to Broadcasted:
            $f(x1::Broadcasted, x2) = broadcast($f, getval(x1), x2) |> Broadcasted
            $f(x1, x2::Broadcasted) = broadcast($f, x1, getval(x2)) |> Broadcasted
            # Ambiguity issues...
            broadcast(::typeof($f), x::Rec) = $bf(x)
            broadcast(::typeof($f), x1::Rec, x2) = $bf(x1,x2)
            broadcast(::typeof($f), x1, x2::Rec) = $bf(x1,x2)
            broadcast(::typeof($f), x1::Rec, x2::Rec) = $bf(x1,x2)
=#

