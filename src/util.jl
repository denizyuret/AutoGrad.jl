# Uncomment to debug:
# macro dbgutil(x); esc(:(println(_dbg($x)))); end
macro dbgutil(x); end

### @primitive and @zerograd macros:

# I would like to make these type signatures as specific as possible.
# The following are not allowed yet, see https://github.com/JuliaLang/julia/issues/3766
# f{T<:Number,A<:AbstractArray{T}}(x::Rec{A})
# f{T<:Number,A<:AbstractArray}(x::Rec{A{T}})

"""

`@primitive fx g1 g2...` can be used to define a new primitive
and (optionally) its gradients.

Julia supports multiple dispatch, i.e. a single function can have
multiple methods with different arg types.  AutoGrad supports
multiple dispatch for primitives and gradients.  Thus fx is a
typed method declaration such as:

* @primitive sin(x::Number)
* @primitive hypot(x1::Array,x2::Array),dy,y

The second example specifies variable names for the output gradient
`dy` and the output `y` after the method declaration which can be used
in gradient expressions.  Untyped, ellipsis and keyword arguments are
ok as in `f(a::Int,b,c...;d=1)`.  Parametric methods such as
`f{T<:Number}(x::T)` cannot be used.

The @primitive macro turns the first example into:

    local sin_r = recorder(sin)
    sin{T<:Number}(x::Rec{T}) = sin_r(x)

This will cause any call to `sin` with a Rec{T<:Number} argument
to be recorded.  With multiple arguments things are a bit more
complicated.  Here is what happens with the second example:

    local hypot_r = recorder(hypot)
    hypot{T<:Array,S<:Array}(x1::Rec{T},x2::Rec{S})=hypot_r(x1,x2)
    hypot{T<:Array,S<:Array}(x1::Rec{T},x2::S)=hypot_r(x1,x2)
    hypot{T<:Array,S<:Array}(x1::T,x2::Rec{S})=hypot_r(x1,x2)

We want the recorder version to be called if any one of the arguments
is a boxed Rec.  There is no easy way to specify this in Julia, so
the macro generates all 2^N-1 boxed/unboxed argument combinations.

The method declaration can optionally be followed by gradient
expressions.  Here are the same examples with gradients:

* @primitive sin(x::Number),dy (dy*cos(x))
* @primitive hypot(x1::Array,x2::Array),dy,y  `(dy.*x1./y)`  `(dy.*x2./y)`

Note that the parameters, the return variable and the output gradient
of the original function can be used in the gradient expressions.

In AutoGrad, gradients are defined using gradient methods that have
the following signature:

    f(Grad{i},dy,y,x...) => dx[i]

For the first example here is the generated gradient method:

`sin{T<:Number}(::Type{Grad{1}}, dy, y, x::Rec{T})=(dy*cos(x))`

For the second example a different gradient method is generated for
each argument:

`hypot{T<:Array,S<:Array}(::Type{Grad{1}},dy,y,x1::Rec{T},x2::Rec{S})=(dy.*x1./y)`
`hypot{T<:Array,S<:Array}(::Type{Grad{2}},dy,y,x1::Rec{T},x2::Rec{S})=(dy.*x2./y)`

In fact @primitive generates four more definitions for the other
boxed/unboxed argument combinations.

Non-differentiable functions such as `sign`, and non-numeric functions
such as `size` should be defined using the @zerograd macro instead.

"""
macro primitive(f,g...)
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
    b = Expr(:block)
    r = gensym()
    push!(b.args, esc(:(local $r = recorder($(fname(f))))))
    rx = rcall(r,f)
    for fx in fsigs(f)
        push!(b.args, esc(:($fx = $rx)))
        for i=1:length(g)
            gx = gsig(fx,dy,y,i)
            push!(b.args, esc(:($gx = $(g[i]))))
        end
    end
    return b
end

"""
`@zerograd f(args...; kwargs...)` allows f to handle its Rec inputs
by unboxing them like @primitive, but unlike @primitive it does not
record its actions or return a Rec result.  Some functions, like
sign(), have zero gradient.  Others, like length() have discrete or
constant outputs.  These need to handle Rec inputs, but do not need
to record anything and can return regular values.  Their output can be
treated like a constant in the program.  Use the @zerograd macro for
those.  Note that kwargs are NOT unboxed.
"""
macro zerograd(f)
    b = Expr(:block)
    f.head == :(::) && (f=f.args[1])
    for fx in fsigs(f)
        zx = zcall(fx)
        push!(b.args, esc(:($fx = $zx)))
    end
    return b
end

function zcall(f)
    z = copy(f)
    z1 = z.args[1]
    isa(z1,Expr) && z1.head==:curly && (z.args[1]=z1.args[1])
    for i=2:length(z.args)
        zi = z.args[i]
        if isa(zi,Symbol)
            # all done
        elseif !isa(zi,Expr)
            error("Unrecognized argtype '$zi'")
        elseif zi.head==:(::)
            (v,t) = zi.args
            if t==:Rec || (isa(t,Expr) && t.head==:curly && t.args[1]==:Rec)
                z.args[i] = :($v.value)
            else
                z.args[i] = v
            end
        elseif zi.head==:(...)  # done
        elseif zi.head==:parameters # done
        else
            error("Unrecognized argtype '$zi'")
        end
    end
    return z
end

# get name out of function declaration
function fname(f)
    n = f.args[1]
    isa(n,Expr) && n.head==:curly && error("parametric methods not currently supported")
    if isa(n,Symbol)
        return n
    else
        error("$n not a symbol")
    end
end

# create call to r using typeless argument of f
function rcall(r,f)
    rx = notypes(f)
    rx.args[1]=r
    # Need to fix kwargs
    r2 = rx.args[2]
    if isa(r2,Expr) && r2.head == :parameters
        for i in 1:length(r2.args)
            k = r2.args[i]
            if !isa(k,Expr); error("Bad kwarg '$k'")
            elseif k.head == :(...); continue
            elseif k.head != :kw; error("Bad kwarg '$k'")
            elseif !isa(k.args[1],Symbol); error("Bad kwarg '$k'")
            else; k.args[2]=k.args[1]; end
        end
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

# create type signatures for f where one or more args are Nodes.
function fsigs(f)
    f1 = copy(f)
    a1 = f1.args[1] = Expr(:curly,fname(f1))
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
        fn = copy(f1)
        iargs = 0
        for i=2:length(fn.args)
            ai = fn.args[i]
            in(ai.head, (:parameters, :(...))) && continue
            ai.head == :(::) || error("Bad arg '$ai'")
            if nodes & (1<<iargs) == 0
                ai.args[2] = Expr(:curly,:Rec,ai.args[2])
            end
            iargs += 1
        end
        push!(flist, fn)
    end
    return flist
end

function gsig(f,dy,y,i)
    g = copy(f)
    if g.args[2].head == :parameters; a = 3; else; a = 2; end
    insert!(g.args, a, :(::Type{Grad{$i}}))
    insert!(g.args, a+1, dy)
    insert!(g.args, a+2, y)
    return g
end


# The way broadcasting works in Julia:
# y = f(x...) where f is a broadcasting operation.
# size(y) = broadcast_shape(x...)
# ndims(y) = max ndims(x)
# size(y,i) = max size(x,i)
# size(x,i) = 1 or size(y,i) for all x and i<=ndims(x)
# if ndims(x) < ndims(y) the extra dimensions of x are treated as 1

function unbroadcast(x, dx)
    if size(x)==size(dx)
        return dx
    elseif isa(getval(x),Number)
        return sum(dx)
    else
        d = []
        for i=1:ndims(dx)
            size(x,i) == size(dx,i) > 1 && continue
            size(x,i) != 1 && throw(DimensionMismatch())
            push!(d,i)
        end
        length(d)==1 && (d=d[1])
        return reshape(sum(dx, d), size(x))
    end
end

function toscalar(xv; rng=MersenneTwister())
    x = getval(xv)
    isa(x,Number) && return xv
    isa(x,OneHot) && (x = full(x))
    idx = isa(x,Tuple) ? (1:length(x)) : eachindex(x)
    s = 0
    for i in idx
        s += xv[i] * rand(rng)
    end
    return s
end

# sumvalues sums values of dictionaries, otherwise acts like sum:

sumvalues(x)=sum(x)
sumvalues(x::Associative)=sum(values(x))
@primitive sumvalues(x::Associative),ds fillvalues(ds,x)
fillvalues(v,x)=(y=similar(x);for k in keys(x); y[k]=v; end; y)
@primitive fillvalues(v,x),dxv sumvalues(dxv) nothing
addtest(sumvalues, Dict(1=>1.,2=>2.))
addtest(fillvalues, 0., Dict(1=>1.,2=>2.,3=>3.))

# This needs more work:
# @primitive values(x),dy Dict(map((a,b)->(a=>b), keys(x), dy))

# It gets tiresome to write `Type{Grad{1}}` after a while, here are
# some convenient aliases:

typealias D1 Type{Grad{1}}
typealias D2 Type{Grad{2}}
if !isdefined(:Dn)
typealias Dn{N} Type{Grad{N}}
end

# Pretty print for debugging:
_dbg(x)=x # extend to define short printable representations
_dbg(x::Tuple)=map(_dbg,x)
_dbg(x::Node)="N$(id2(x))_$(id2(x.rec))"
_dbg(x::Rec)="V$(id2(x))_$(_dbg(x.value))"
_dbg(x::Tape)="T$(join([id2(x),map(id2,x)...],'_'))"
_dbg(x::AbstractArray)="A$(join([id2(x),size(x)...],'_'))"
id2(x)=Int(object_id(x)%1000)

Base.show(io::IO, n::Rec) = print(io, _dbg(n))
Base.show(io::IO, n::Node) = print(io, _dbg(n))
Base.show(io::IO, n::Tape) = print(io, _dbg(n))

