# Uncomment to debug:
# macro dbgutil(x); esc(:(println(_dbg($x)))); end
macro dbgutil(x); end

### @primitive and @zerograd macros:

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
    sin{T<:Number}(x::Value{T}) = sin_r(x)

This will cause any call to `sin` with a Value{T<:Number} argument
to be recorded.  With multiple arguments things are a bit more
complicated.  Here is what happens with the second example:

    local hypot_r = recorder(hypot)
    hypot{T<:Array,S<:Array}(x1::Value{T},x2::Value{S})=hypot_r(x1,x2)
    hypot{T<:Array,S<:Array}(x1::Value{T},x2::S)=hypot_r(x1,x2)
    hypot{T<:Array,S<:Array}(x1::T,x2::Value{S})=hypot_r(x1,x2)

We want the recorder version to be called if any one of the arguments
is a boxed Value.  There is no easy way to specify this in Julia, so
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

`sin{T<:Number}(::Type{Grad{1}}, dy, y, x::Value{T})=(dy*cos(x))`

For the second example a different gradient method is generated for
each argument:

`hypot{T<:Array,S<:Array}(::Type{Grad{1}},dy,y,x1::Value{T},x2::Value{S})=(dy.*x1./y)`
`hypot{T<:Array,S<:Array}(::Type{Grad{2}},dy,y,x1::Value{T},x2::Value{S})=(dy.*x2./y)`

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
    addtest(f)
    return b
end

"""
`@zerograd f(args...; kwargs...)` allows f to handle its Value inputs
by unboxing them like @primitive, but unlike @primitive it does not
record its actions or return a Value result.  Some functions, like
sign(), have zero gradient.  Others, like length() have discrete or
constant outputs.  These need to handle Value inputs, but do not need
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
            if t==:Value || (isa(t,Expr) && t.head==:curly && t.args[1]==:Value)
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
    isa(n,Symbol) || error("Function name $n not a symbol")
    return n
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
            !isa(k,Expr) ? error("Bad kwarg '$k'") :
            k.head == :(...) ? continue :
            k.head != :kw ? error("Bad kwarg '$k'") :
            !isa(k.args[1],Symbol) ? error("Bad kwarg '$k'") :
            k.args[2]=k.args[1]
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
                ai.args[2] = Expr(:curly,:Value,ai.args[2])
            end
            iargs += 1
        end
        push!(flist, fn)
    end
    return flist
end

function gsig(f,dy,y,i)
    g = copy(f)
    a = (g.args[2].head == :parameters ? 3 : 2)
    insert!(g.args, a, :(::Type{Grad{$i}}))
    insert!(g.args, a+1, dy)
    insert!(g.args, a+2, y)
    return g
end



### Testing Utilities:

if !isdefined(:runtests)
let tests=[]
    global addtest,runtests,alltests
    alltests()=tests
    addtest(t)=push!(tests,t)
    function runtests(a=tests)
        for fx in a
            tx = fixtest(fx)
            try 
                check_grads(tx...; fname=fx.args[1])
            catch e
                warn((fx,tx[2:end]...,"$e"))
            end
        end
    end
end
end

function fixtest(fx::Expr)
    # get the function
    fx.head == :call || error("Expecting function declaration got '$fx'")
    fname = fx.args[1]
    isa(fname,Symbol) || error("$fname not a Symbol")
    f = eval(fname)
    # prepare arguments
    x = Any[]
    for i=2:length(fx.args)
        ai = fx.args[i]
        isa(ai,Symbol) ? push!(x,randn(2)) :
        !isa(ai,Expr) ? error("Neither Symbol nor Expr: $ai") :
        ai.head == :parameters ? nothing :
        ai.head == :(...) ? nothing :
        ai.head != :(::) ? error("Argtype not supported: '$ai'") :
        ai.args[2] == :Number ? push!(x,randn()) :
        ai.args[2] == :AbstractFloat ? push!(x,randn()) :
        ai.args[2] == :AbstractArray ? push!(x,randn(2)) :
        ai.args[2] == :AbstractVecOrMat ? push!(x,rand()<0.5 ? randn(2) : randn(2,2)) :
        # ai.args[2] == :AorN ? push!(x,rand()<0.5 ? randn() : randn(2)) :
        ai.args[2] == :Associative ? push!(x,Dict()) :
        ai.args[2] == :Tuple ? push!(x,()) :
        (warn("Don't know how to sample $(ai.args[2])"); push!(x,nothing))
    end
    # fix the arguments to be in the right domain for f
    x = fixdomain(Val{fname},x...)
    y = f(x...)
    # detect and prevent testing of zero / undefined grads
    plist = Any[]
    alist = Any[x...]           # define fnew(plist)=f(alist)
    args = Any[]                   # call fnew(args...)
    gargs = (Value(y), Value(y), map(Value,x)...)
    for i=1:length(alist)
        g = 0
        try
            g = f(Grad{i},gargs...)
        catch e
            # warn("No grad $i for $f: $e")
            continue            # undefined grads
        end
        g == 0 && continue      # zero grads
        push!(args, alist[i])
        alist[i] = Symbol("x$i")
        push!(plist, alist[i])
    end
    f1=f; f = eval(Expr(:->, Expr(:tuple, plist...), Expr(:call, f1, alist...)))
    # if f has non-scalar output, sum it
    isa(f(args...),Number) || (f2=f; f=(x...)->sum(f2(x...)))
    return (f,args...)
end

# Override this for testing restricted domain functions like acos:
fixdomain(f,x...)=x

if !isdefined(:Fn)
typealias Fn{F} Type{Val{F}}    # used to create the first argument of fixdomain
end                             # e.g. fixdomain(::Fn{:log},x)=abs(x)


# I would like to make these type signatures as specific as possible.
# The following are not allowed yet, see https://github.com/JuliaLang/julia/issues/3766
# f{T<:Number,A<:AbstractArray{T}}(x::Value{A})
# f{T<:Number,A<:AbstractArray}(x::Value{A{T}})


EPS, RTOL, ATOL = 1e-4, 1e-4, 1e-6

# TODO: do sampling or random direction for large args
"""
check_grads(fun, args...) checks the computed gradients for fun(args)
comparing them with numeric approximations.
"""
function check_grads(fun, args...; eps=EPS, rtol=RTOL, atol=ATOL, fname=fun)
    @dbgutil((:check_grads,fname,:args,args...))
    isempty(args) && error("No args given")
    exact = ntuple(i->grad(fun,i)(args...), length(args))
    numeric = nd(fun, args...; eps=eps)
    @dbgutil((:check_grads,fname,:exact,exact,:numeric,numeric))
    same = isequivalent(exact, numeric; rtol=rtol, atol=atol)
    same || warn((:check_grads,fname,:args,args,:exact,exact,:numeric,numeric))
    return same
end

function nd(f, args...; eps=EPS)
    @dbgutil((:nd,f,args..., :eps, eps))
    unary_f = x->f(x...)
    unary_nd(unary_f, tofloat(args), eps)
end

unary_nd(f, x::Tuple, eps)         = ntuple(i->unary_nd(indexed_function(f, x, i), x[i], eps), length(x))
unary_nd(f, x::Associative, eps)   = (a=similar(x); for(k,v) in x; a[k] = unary_nd(indexed_function(f, x, k), v, eps); end; a)
unary_nd(f, x::AbstractArray, eps) = reshape(eltype(x)[unary_nd(indexed_function(f, x, i), v, eps) for (i,v) in enumerate(x)], size(x))
unary_nd(f, x::Complex, eps)       = ((f(x + eps/2) - f(x - eps/2)) / eps - im*(f(x + im*eps/2) - f(x - im*eps/2)) / eps)
unary_nd(f, x::Real, eps)          = ((f(x + eps/2) - f(x - eps/2)) / eps)

function indexed_function(fun, arg, index)
    function partial_function(x)
        if isa(arg, Tuple)
            local_arg = (arg[1:index-1]..., x, arg[index+1:end]...)
        else
            local_arg = copy(arg); local_arg[index] = x
        end
        return fun(local_arg)
    end
    return partial_function
end

# isequivalent uses isapprox for Number and AbstractArray{T<:Number}
isequivalent(x::Number,y::Number; o...)=isapprox(x,y;o...)
isequivalent{T<:Number,S<:Number}(x::AbstractArray{T},y::AbstractArray{S}; o...)=isapprox(x,y;o...)

# isequivalent extends to Tuple, Associative, and other Arrays, comparing elementwise
isequivalent(x::Tuple, y::Tuple; o...)=(length(x)==length(y) && all(i->isequivalent(x[i],y[i];o...), 1:length(x)))
isequivalent(x::Associative, y::Associative; o...)=(length(x)==length(y) && all(k->isequivalent(x[k],y[k];o...), keys(x)))
isequivalent(x::AbstractArray, y::AbstractArray; o...)=(length(x)==length(y) && all(i->isequivalent(x[i],y[i];o...), 1:length(x)))

# isequivalent treats `nothing` as equivalent to zero or zero array.
isequivalent(x::Number,z::Void; o...)=isequivalent(z,x;o...)
isequivalent{T<:Number}(x::AbstractArray{T},z::Void; o...)=isequivalent(z,x;o...)
isequivalent(z::Void,x::Number; o...)=isapprox(zero(x),x;o...)
isequivalent{T<:Number}(z::Void,x::AbstractArray{T}; rtol::Real=Base.rtoldefault(T), atol::Real=0, norm::Function=vecnorm) = (norm(x) <= atol/(1-rtol)) # Modified from: linalg/generic.jl:522

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
            size(x,i) == size(dx,i) && continue
            size(x,i) != 1 && throw(DimensionMismatch())
            push!(d,i)
        end
        length(d)==1 && (d=d[1])
        return sum(dx, d)
    end
end

# findfirst uses == which is inefficient for tapes, so we define findeq with ===
function findeq(A,v)
    for i=1:length(A)
        if A[i] === v
            return i
        end
    end
    return 0
end

# typealias AorN Union{AbstractArray,Number}

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
_dbg(x::Node)=Symbol("N$(id2(x))_$(id2(x.value))")
_dbg(x::Value)=Symbol("V$(id2(x))_$(id2(x.value))")
_dbg(x::Tape)=Symbol("T$(join([id2(x),map(id2,x)...],'_'))")
_dbg(x::AbstractArray)=Symbol("A$(join([id2(x),size(x)...],'_'))")
id2(x)=Int(object_id(x)%1000)

Base.show(io::IO, n::Value) = print(io, _dbg(n))
Base.show(io::IO, n::Node) = print(io, _dbg(n))
Base.show(io::IO, n::Tape) = print(io, _dbg(n))

# TODO: check if we really need tofloat.
# converts nested values to float.
# tofloat uses float for Number and AbstractArray (for isleaftype)
# tofloat extends to Tuple, Associative, and arbitrary Arrays

isfloat(x)=false
isfloat(::AbstractFloat)=true
isfloat{T<:AbstractFloat}(::AbstractArray{T})=true
isfloat{T}(x::AbstractArray{T})=(!isbits(T) && all(isfloat,x))
isfloat(x::Tuple)=all(isfloat,x)
isfloat(x::Associative)=all(isfloat,values(x))

tofloat(x)=float(x)
tofloat(x::AbstractFloat)=x
tofloat{T<:AbstractFloat}(x::AbstractArray{T})=x
tofloat{T}(x::AbstractArray{T})=(isfloat(x) ? x : isbits(T) ? float(x) : map(tofloat,x))
tofloat(x::Tuple)=(isfloat(x) ? x : ntuple(i->tofloat(x[i]), length(x)))
tofloat(x::Associative)=(isfloat(x) ? x : (a=similar(x); for (k,v) in x; a[k]=tofloat(v); end; a))
