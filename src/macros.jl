"""

    @primitive  fx g1 g2...

Define a new primitive operation for AutoGrad and (optionally) specify its gradients.
Non-differentiable functions such as `sign`, and non-numeric functions such as `size` should
be defined using the @zerograd macro instead.

# Examples

    @primitive sin(x::Number)
    @primitive hypot(x1,x2),dy,y

    @primitive sin(x::Number),dy  (dy.*cos(x))
    @primitive hypot(x1,x2),dy,y  (dy.*x1./y)  (dy.*x2./y)

The first example shows that `fx` is a typed method declaration.  Julia supports multiple
dispatch, i.e. a single function can have multiple methods with different arg types.
AutoGrad takes advantage of this and supports multiple dispatch for primitives and
gradients.

The second example specifies variable names for the output gradient `dy` and the output `y`
after the method declaration which can be used in gradient expressions.  Untyped, ellipsis
and keyword arguments are ok as in `f(a::Int,b,c...;d=1)`.  Parametric methods such as
`f(x::T) where {T<:Number}` cannot be used.

The method declaration can optionally be followed by gradient expressions.  The third and
fourth examples show how gradients can be specified.  Note that the parameters, the return
variable and the output gradient of the original function can be used in the gradient
expressions.

# Under the hood

The @primitive macro turns the first example into:

    sin(x::Value{T}) where {T<:Number} = forw(sin, x)

This will cause calls to `sin` with a boxed argument (`Value{T<:Number}`) to be recorded.
The recorded operations are used by AutoGrad to construct a dynamic computational graph.
With multiple arguments things are a bit more complicated.  Here is what happens with the
second example:

    hypot(x1::Value{S}, x2::Value{T}) where {S,T} = forw(hypot, x1, x2)
    hypot(x1::S, x2::Value{T})        where {S,T} = forw(hypot, x1, x2)
    hypot(x1::Value{S}, x2::T)        where {S,T} = forw(hypot, x1, x2)

We want the forw method to be called if any one of the arguments is a boxed `Value`.  There
is no easy way to specify this in Julia, so the macro generates all 2^N-1 boxed/unboxed
argument combinations.

In AutoGrad, gradients are defined using gradient methods that have the following pattern:

    back(f,Arg{i},dy,y,x...) => dx[i]

For the third example here is the generated gradient method:

    back(::typeof(sin), ::Type{Arg{1}}, dy, y, x::Value{T}) where {T<:Number} = dy .* cos(x)

For the last example a different gradient method is generated for each argument:

    back(::typeof(hypot), ::Type{Arg{1}}, dy, y, x1::Value{S}, x2::Value{T}) where {S,T} = (dy .* x1) ./ y
    back(::typeof(hypot), ::Type{Arg{2}}, dy, y, x1::Value{S}, x2::Value{T}) where {S,T} = (dy .* x2) ./ y

In fact @primitive generates four more definitions for the other boxed/unboxed argument
combinations.

# Broadcasting

Broadcasting is handled by extra `forw` and `back` methods. `@primitive` defines the following 
so that broadcasting of a primitive function with a boxed value triggers `forw` and `back`.

    broadcasted(::typeof(sin), x::Value{T}) where {T<:Number} = forw(broadcasted,sin,x)
    back(::typeof(broadcasted), ::Type{Arg{2}}, dy, y, ::typeof(sin), x::Value{T}) where {T<:Number} = dy .* cos(x)

If you do not want the broadcasting methods, you can use the `@primitive1` macro. If you
only want the broadcasting methods use `@primitive2`. As a motivating example, here is how
`*` is defined for non-scalars:

    @primitive1 *(x1,x2),dy  (dy*x2')  (x1'*dy)
    @primitive2 *(x1,x2),dy  unbroadcast(x1,dy.*x2)  unbroadcast(x2,x1.*dy)

Regular `*` is matrix multiplication, broadcasted `*` is elementwise multiplication and the
two have different gradients as defined above. `unbroadcast(a,b)` reduces `b` to the same
shape as `a` by performing the necessary summations.
"""
:(@primitive), :(@primitive1), :(@primitive2)

macro primitive(f,g...)                         # @primitive sin(x::Number),dy,y  (dy.*cos.(x))
    (f,dy,y) = fparse(f)
    b = Expr(:block)
    forwcall = fcall(f)                	    	# forw(sin,x)
    forwcast = fcall(f,broadcasted=true)        # forw(broadcasted,sin,x)
    for fx in fsigs(f)                          # sin(x::Value{T}) where {T<:Number}
        push!(b.args, :($fx = $forwcall))
        bfx = f2b(fx)                           # broadcasted(::typeof(sin), x::Value{T}) where {T<:Number}
        push!(b.args, :($bfx = $forwcast))
        for i=1:length(g)
            gx = gsig(fx,dy,y,i)                # back(::typeof(sin), ::Type{Arg{1}}, dy, y, x::Value{T}) where {T<:Number}
            push!(b.args, :($gx = $(g[i])))     # '' = (dy.*cos.(x))
            bgx = bsig(fx,dy,y,i)               # back(::typeof(broadcasted), ::Type{Arg{2}}, dy, y, ::typeof(sin), x::Value{T}) where {T<:Number}
            push!(b.args, :($bgx = $(g[i])))
        end
    end
    return esc(b)
end

macro primitive1(f,g...)        # non-broadcasting version
    (f,dy,y) = fparse(f)
    b = Expr(:block)
    forwcall = fcall(f)
    for fx in fsigs(f)
        push!(b.args, :($fx = $forwcall))
        for i=1:length(g)
            gx = gsig(fx,dy,y,i)
            push!(b.args, :($gx = $(g[i])))
        end
    end
    return esc(b)
end

macro primitive2(f,g...)        # broadcasting-only version
    (f,dy,y) = fparse(f) 
    b = Expr(:block) 
    forwcast = fcall(f,broadcasted=true) 
    for fx in fsigs(f) 
        bfx = f2b(fx) 
        push!(b.args, :($bfx = $forwcast)) 
        for i=1:length(g) 
            bgx = bsig(fx,dy,y,i)
            push!(b.args, :($bgx = $(g[i])))
        end
    end
    return esc(b)
end

"""

    @zerograd f(args...; kwargs...)

Define `f` as an AutoGrad primitive operation with zero gradient.
    
# Example:

    @zerograd  floor(x::Float32)

`@zerograd` allows `f` to handle boxed `Value` inputs by unboxing them like a `@primitive`,
but unlike `@primitive` it does not record its actions or return a boxed `Value` result.
Some functions, like `sign()`, have zero gradient.  Others, like `length()` have discrete or
constant outputs.  These need to handle `Value` inputs, but do not need to record anything
and can return regular values.  Their output can be treated like a constant in the program.
Use the `@zerograd` macro for those.  Use the `@zerograd1` variant if you don't want to
define the broadcasting version and `@zerograd2` if you only want to define the broadcasting
version. Note that `kwargs` are NOT unboxed.

"""
:(@zerograd), :(@zerograd1), :(@zerograd2)

macro zerograd(f)                               # @zerograd sign(x::Number)
    (f,dy,y) = fparse(f)
    b = Expr(:block)
    for fx in fsigs(f)                          # sign(x::Value{T}) where {T<:Number}
        zx = zcall(fx)                          # sign(value(x))
        push!(b.args, esc(:($fx = $zx)))
        (bfx,bzx) = bzcall(fx,zx)
        push!(b.args, esc(:($bfx = $bzx)))      # broadcasted(::typeof(sign), x::Value{T}) where {T <: Number} = broadcasted(sign, value(x))
        bx = v2b(fx)                            # sign(x::Bcasted{T}) where {T<:Number}
        push!(b.args, esc(:($bx = AutoGrad.Bcasted($bzx)))) # ... = Bcasted(broadcasted(sign, value(x)))
    end
    return b
end

macro zerograd1(f)   # non-broadcasting version
    (f,dy,y) = fparse(f)
    b = Expr(:block)
    for fx in fsigs(f)
        zx = zcall(fx)
        push!(b.args, esc(:($fx = $zx)))
    end
    return b
end

macro zerograd2(f)   # broadcasting-only version
    (f,dy,y) = fparse(f)
    b = Expr(:block)
    for fx in fsigs(f)
        zx = zcall(fx)
        #push!(b.args, esc(:($fx = $zx)))
        (bfx,bzx) = bzcall(fx,zx)
        push!(b.args, esc(:($bfx = $bzx)))
        bx = v2b(fx)
        push!(b.args, esc(:($bx = AutoGrad.Bcasted($bzx))))
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

function fcall(f; broadcasted = false)
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
        a = 3
    else
        a = 2
    end
    insert!(rx.args,a,fn)
    if broadcasted; insert!(rx.args,a,:(Base.Broadcast.broadcasted)); end
    return rx
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
    bzx.args[1] = :(Base.Broadcast.broadcasted)
    insert!(bzx.args, a, fname)
    return (bfx,bzx)
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

# input: (where (call f (:: x (curly Value T))) (<: T Int))
# output: (where (call broadcasted (:: (call typeof f)) (:: x (curly Value T))) (<: T Int))
function f2b(fx)
    bx = copy(fx)               # where...
    cx = bx.args[1]             # call...
    f = cx.args[1]              # func
    cx.args[1] = :(Base.Broadcast.broadcasted)
    if cx.args[2].head == :parameters; a = 3; else; a = 2; end
    insert!(cx.args, a, :(::typeof($f)))
    return bx
end

# change AutoGrad.Value -> AutoGrad.Bcasted
function v2b(fx)
    if fx == :(AutoGrad.Value)
        :(AutoGrad.Bcasted)
    elseif isa(fx, Expr)
        Expr(fx.head, v2b.(fx.args)...)
    else
        fx
    end
end

# create type signatures for f where one or more args are Value's.
# With multiple args add Value to each subset combinatorially.
# The input has the form (call f (:: x Int))
# The 0.6 output was     (call (curly f (<: T Int)) (:: x (curly Value T)))
# The 0.7 output is      (where (call f (:: x (curly Value T))) (<: T Int))
function fsigs(f)               # sin(x::Number) => sin(x::Value{T}) where {T <: Number}
    f1 = copy(f)                # sin(x::Number)
    fname = f1.args[1]
    if isa(fname, Symbol) && isdefined(@__MODULE__, fname) # @__MODULE__ here resolves to AutoGrad when compiling Knet
        m = which(@__MODULE__, fname)                      # which does not work for symbols undefined in AutoGrad
        f1.args[1] = :(($m).$fname) # Base.sin
    end
    a1 = Expr(:where,f1)        # sin(x::Number) where {}
    nargs = 0
    for i=2:length(f1.args)
        ai = f1.args[i]         # x::Number
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
        fn = copy(a1)           # sin(x::T) where {T <: Number}
        f1 = fn.args[1]         # sin(x::T)
        iargs = 0
        for i=2:length(f1.args)
            ai = f1.args[i]     # (x::T)
            in(ai.head, (:parameters, :(...))) && continue
            ai.head == :(::) || error("Bad arg '$ai'")
            if nodes & (1<<iargs) == 0
                ai.args[2] = :(AutoGrad.Value{$(ai.args[2])}) #Expr(:curly,:Value,ai.args[2])
            end
            iargs += 1
        end
        push!(flist, fn)        # sin(x::Value{T}) where {T <: Number}
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
    insert!(g.args, a+1, :(::Type{AutoGrad.Arg{$i}}))
    insert!(g.args, a+2, dy)
    insert!(g.args, a+3, y)
    return fcopy
end

# This is for the broadcast version
# Input: (where (call f (:: x (curly Value T))) (<: T Int))
# Output: (where (call broadcasted :(::Type{Grad{2}}) dy y :(::typeof(f)) :(x::Value{T})) (<: T Int))
function bsig(f,dy,y,i)
    fcopy = copy(f)
    g = fcopy.args[1]
    fname = g.args[1]
    g.args[1] = :(AutoGrad.back)
    if g.args[2].head == :parameters; a = 3; else; a = 2; end
    insert!(g.args, a, :(::typeof(Base.Broadcast.broadcasted)))
    insert!(g.args, a+1, :(::Type{AutoGrad.Arg{$(i+1)}}))
    insert!(g.args, a+2, dy)
    insert!(g.args, a+3, y)
    insert!(g.args, a+4, :(::typeof($fname)))
    return fcopy
end


# I would like to make these type signatures as specific as possible.
# The following are not allowed yet, see https://github.com/JuliaLang/julia/issues/3766
# f{T<:Number,A<:AbstractArray{T}}(x::Value{A})
# f{T<:Number,A<:AbstractArray}(x::Value{A{T}})
# 20180725: TODO: This may have changed in Julia 0.7

