### @primitive and @zerograd macros:

# I would like to make these type signatures as specific as possible.
# The following are not allowed yet, see https://github.com/JuliaLang/julia/issues/3766
# f{T<:Number,A<:AbstractArray{T}}(x::Value{A})
# f{T<:Number,A<:AbstractArray}(x::Value{A{T}})
# 20180725: TODO: This may have changed in Julia 0.7

"""

    @primitive  fx g1 g2...

Define a new primitive operation for AutoGrad and (optionally) specify
its gradients.  Non-differentiable functions such as `sign`, and
non-numeric functions such as `size` should be defined using the
@zerograd macro instead.

# Examples

    @primitive sin(x::Number)
    @primitive hypot(x1,x2),dy,y

    @primitive sin(x::Number),dy  (dy.*cos(x))
    @primitive hypot(x1,x2),dy,y  (dy.*x1./y)  (dy.*x2./y)

The first example shows that `fx` is a typed method declaration.
Julia supports multiple dispatch, i.e. a single function can have
multiple methods with different arg types.  AutoGrad takes advantage
of this and supports multiple dispatch for primitives and gradients.

The second example specifies variable names for the output gradient
`dy` and the output `y` after the method declaration which can be used
in gradient expressions.  Untyped, ellipsis and keyword arguments are
ok as in `f(a::Int,b,c...;d=1)`.  Parametric methods such as
`f(x::T) where {T<:Number}` cannot be used.

The method declaration can optionally be followed by gradient
expressions.  The third and fourth examples show how gradients can be
specified.  Note that the parameters, the return variable and the
output gradient of the original function can be used in the gradient
expressions.

# Under the hood

The @primitive macro turns the first example into:

    sin(x::Value{T}) where {T<:Number} = forw(sin, x)

This will cause calls to `sin` with a boxed argument
(`Value{T<:Number}`) to be recorded.  The recorded operations are used
by AutoGrad to construct a dynamic computational graph.  With multiple
arguments things are a bit more complicated.  Here is what happens
with the second example:

    hypot(x1::Value{S}, x2::Value{T}) where {S<:Any,T<:Any} = forw(hypot, x1, x2)
    hypot(x1::S, x2::Value{T})      where {S<:Any,T<:Any} = forw(hypot, x1, x2)
    hypot(x1::Value{S}, x2::T)      where {S<:Any,T<:Any} = forw(hypot, x1, x2)

We want the forw method to be called if any one of the arguments
is a boxed `Value`.  There is no easy way to specify this in Julia, so
the macro generates all 2^N-1 boxed/unboxed argument combinations.

In AutoGrad, gradients are defined using gradient methods that have
the following pattern:

    back(f,Val(i),dy,y,x...) => dx[i]

For the third example here is the generated gradient method:

    back(::typeof(sin), ::Val{1}, dy, y, x::Value{T}) where {T<:Number} = dy .* cos(x)

For the last example a different gradient method is generated for each
argument:

    back(::typeof(hypot), ::Val{1}, dy, y, x1::Value{S}, x2::Value{T}) where {S<:Any,T<:Any} = (dy .* x1) ./ y
    back(::typeof(hypot), ::Val{2}, dy, y, x1::Value{S}, x2::Value{T}) where {S<:Any,T<:Any} = (dy .* x2) ./ y

In fact @primitive generates four more definitions for the other
boxed/unboxed argument combinations.

# Broadcasting

Broadcasting is handled by extra `forw` and `back` methods. In `broadcast.jl` we define:

    broadcasted(f, x::Value) = forw(broadcast,f,x)

and similar methods that match any function `f`, so that when a boxed
value is in a broadcasting operation `forw` is called. The
`@primitive` macro defines the `back` method for broadcasting of a
particular primitive:

    back(::typeof(broadcast), ::Val{2}, dy, y, ::typeof(sin), x::Value{T}) where {T<:Number} = dy .* cos(x)

If you do not want the back method for broadcasting, you can use the
`@primitive1` macro which omits this final definition.

"""
macro primitive(f,g...)
    (f,dy,y) = fparse(f)
    b = Expr(:block)
    rx = fcall(f)             # e.g. forw(sin,x)
    for fx in fsigs(f)
        push!(b.args, :($fx = $rx)) # e.g. sin(x::Value{T}) where {T<:Number} = forw(sin,x)
        for i=1:length(g)
            gx = gsig(fx,dy,y,i)
            push!(b.args, :($gx = $(g[i]))) # e.g. back(::typeof(sin), ::Val{1}, dy, y, x::Value{T}) where {T<:Number} = (dy.*cos.(x))
            bx = bsig(fx,dy,y,i)
            push!(b.args, :($bx = $(g[i]))) # e.g. back(::typeof(broadcast), ::Val{2}, dy, y, ::typeof(sin), x::Value) = (dy.*cos.(x))
        end
    end
    return esc(b)
end

# Do we need the version without broadcasting?
macro primitive1(f,g...)
    (f,dy,y) = fparse(f)
    b = Expr(:block)
    rx = fcall(f)
    for fx in fsigs(f)
        push!(b.args, :($fx = $rx)) # e.g. sin(x::Value{T}) where {T<:Number} = sin_r(x)
        for i=1:length(g)
            gx = gsig(fx,dy,y,i)
            push!(b.args, :($gx = $(g[i]))) # e.g. sin(::Type{Grad{1}}, dy, y, x::Value{T}) where {T<:Number} = (dy.*cos.(x))
        end
    end
    return esc(b)
end


"""

    @zerograd f(args...; kwargs...)

Define `f` as an AutoGrad primitive operation with zero gradient.
    
# Example:

    @zerograd  floor(x::Float32)

`@zerograd` allows `f` to handle boxed `Value` inputs by unboxing them
like a `@primitive`, but unlike `@primitive` it does not record its
actions or return a boxed `Value` result.  Some functions, like
`sign()`, have zero gradient.  Others, like `length()` have discrete
or constant outputs.  These need to handle `Value` inputs, but do not
need to record anything and can return regular values.  Their output
can be treated like a constant in the program.  Use the `@zerograd`
macro for those.  Use the `@zerograd1` variant if you don't want to
define the broadcasting version. Note that `kwargs` are NOT unboxed.

"""
macro zerograd(f)
    f.head == :(::) && (f=f.args[1])
    f.head == :call || error("'$f' not a method signature")
    b = Expr(:block)
    for fx in fsigs(f)          # e.g. sign(x::Value{T}) where {T<:Number}
        zx = zcall(fx)          # e.g. sign(value(x))
        push!(b.args, esc(:($fx = $zx)))
        (bfx,bzx) = bzcall(fx,zx)
        push!(b.args, esc(:($bfx = $bzx))) # e.g. broadcast(::typeof(sign), x::Value{T}) where T <: Any) = broadcast(sign, value(x))
    end
    return b
end

# Do we need the version without broadcasting?
macro zerograd1(f)
    f.head == :(::) && (f=f.args[1])
    f.head == :call || error("'$f' not a method signature")
    b = Expr(:block)
    for fx in fsigs(f)          # e.g. sign(x::Value{T}) where {T<:Number}
        zx = zcall(fx)          # e.g. sign(value(x))
        push!(b.args, esc(:($fx = $zx)))
    end
    return b
end

function fparse(f)
    isa(f,Expr) || error("'$f' not a method signature")
    if f.head == :tuple # Using f(x),dy,y to indicate return variable for gradients
        if length(f.args) == 3
            (f,dy,y) = f.args
        elseif length(f.args) == 2
            (f,dy) = f.args; y = gensym()
        else
            error("The first arg '$f' should have the format f(x),dy,y")
        end
    else
        dy = gensym(); y = gensym()
    end
    f.head == :call || error("'$f' not a method signature")
    isa(dy,Symbol) || error("Output gradient '$dy' not a symbol")
    isa(y,Symbol) || error("Return variable '$y' not a symbol")
    return (f,dy,y)
end    

# Input is of the form: (where (call f (:: x (curly (. AutoGrad Value) T))) (<: T Int))
function zcall(f)
    z = copy(f.args[1])
    z1 = z.args[1]
    isa(z1,Expr) && z1.head==:curly && (z.args[1]=z1.args[1]) # This should not be needed in 0.7
    for i=2:length(z.args)
        zi = z.args[i]
        if isa(zi,Symbol)
            # all done
        elseif !isa(zi,Expr)
            error("Unrecognized argtype '$zi'")
        elseif zi.head==:(::)
            (v,t) = zi.args
            if t==:(AutoGrad.Value) || (isa(t,Expr) && t.head==:curly && t.args[1]==:(AutoGrad.Value))
                z.args[i] = :(value($v))
            else
                z.args[i] = v
            end
        elseif zi.head==:(...)  # done
        elseif zi.head==:parameters # done
        else
            error("Unrecognized argtype '$zi'")
        end
    end
    return notypes(z)
end

function bzcall(fx,zx)
    bfx = copy(fx)
    g = bfx.args[1]
    fname = g.args[1]
    g.args[1] = :(Base.Broadcast.broadcasted)
    if g.args[2].head == :parameters; a = 3; else; a = 2; end
    insert!(g.args, a, :(::typeof($fname)))
    bzx = copy(zx)
    bzx.args[1] = :broadcasted
    insert!(bzx.args, a, fname)
    return (bfx,bzx)
end

function fcall(f)
    rx = notypes(f)
    fn = rx.args[1]
    rx.args[1]=:(AutoGrad.forw)
    # Need to fix kwargs
    r2 = rx.args[2]
    if isa(r2,Expr) && r2.head == :parameters
        for i in 1:length(r2.args)
            k = r2.args[i]
            if isa(k,Symbol); r2.args[i] = Expr(:kw,k,k)
            elseif !isa(k,Expr); error("Bad kwarg '$k'")
            elseif k.head == :(...); continue
            elseif k.head != :kw; error("Bad kwarg '$k'")
            elseif !isa(k.args[1],Symbol); error("Bad kwarg '$k'")
            else; k.args[2]=k.args[1]; end
        end
        insert!(rx.args,3,fn)
    else
        insert!(rx.args,2,fn)
    end
    return rx
end

# eliminate type declarations from a function call
function notypes(ex)
    if isa(ex, Expr)
        if (ex.head == :(::) || ex.head == :curly)
            return notypes(ex.args[1])
        else
            return Expr(ex.head, map(notypes, ex.args)...)
        end
    else
        return ex
    end
end

# create type signatures for f where one or more args are Value's.
# With multiple args add Value to each subset combinatorially.
# The input has the form (call f (:: x Int))
# The 0.6 output was     (call (curly f (<: T Int)) (:: x (curly Value T)))
# The 0.7 output is      (where (call f (:: x (curly Value T))) (<: T Int))
function fsigs(f)
    f1 = copy(f)
    a1 = Expr(:where,f1)
    nargs = 0
    for i=2:length(f1.args)
        ai = f1.args[i]
        if isa(ai,Symbol)
            nargs+=1
            ti = gensym()
            push!(a1.args, Expr(:<:, ti, Any))
            f1.args[i] = Expr(:(::),ai,ti)
        elseif !isa(ai,Expr)
            error("Neither Symbol nor Expr: $ai")
        elseif in(ai.head, (:parameters, :(...)))
            continue
        elseif ai.head == :(::)
            nargs+=1
            ti = gensym()
            push!(a1.args, Expr(:<:,ti,ai.args[2]))
            ai.args[2] = ti
        else
            error("Argtype not supported: '$ai'")
        end
    end
    flist = []
    for nodes=0:(1<<nargs-2)
        fn = copy(a1)
        f1 = fn.args[1]
        iargs = 0
        for i=2:length(f1.args)
            ai = f1.args[i]
            in(ai.head, (:parameters, :(...))) && continue
            ai.head == :(::) || error("Bad arg '$ai'")
            if nodes & (1<<iargs) == 0
                ai.args[2] = :(AutoGrad.Value{$(ai.args[2])}) #Expr(:curly,:Value,ai.args[2])
            end
            iargs += 1
        end
        push!(flist, fn)
    end
    return flist
end

# The first input to gsig is an output of fsigs, e.g.
# (where (call f (:: x (curly Value T))) (<: T Int))
function gsig(f,dy,y,i)
    fcopy = copy(f)
    g = fcopy.args[1]
    fname = g.args[1]
    g.args[1] = :(AutoGrad.back)
    if g.args[2].head == :parameters; a = 3; else; a = 2; end
    insert!(g.args, a, :(::typeof($fname)))
    insert!(g.args, a+1, :(::Val{$i}))
    insert!(g.args, a+2, dy)
    insert!(g.args, a+3, y)
    return fcopy
end

# This is for the broadcast version
# Input: (where (call f (:: x (curly Value T))) (<: T Int))
# Output: (where (call broadcast :(::Type{Grad{2}}) dy y :(::typeof(f)) :(x::Value{T})) (<: T Int))
function bsig(f,dy,y,i)
    fcopy = copy(f)
    g = fcopy.args[1]
    fname = g.args[1]
    g.args[1] = :(AutoGrad.back)
    if g.args[2].head == :parameters; a = 3; else; a = 2; end
    insert!(g.args, a, :(::typeof(broadcast)))
    insert!(g.args, a+1, :(::Val{$(i+1)}))
    insert!(g.args, a+2, dy)
    insert!(g.args, a+3, y)
    insert!(g.args, a+4, :(::typeof($fname)))
    return fcopy
end

