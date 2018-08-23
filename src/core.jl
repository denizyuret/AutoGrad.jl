# Contents:

# Here are the rough steps performed by g(x) where g=grad(f):
# 1. g is called with the same inputs as f.
# 2. g calls forward_pass which boxes x in a Rec type and calls f(Rec(x)).
# 3. If a primitive operator inside f gets a Rec input, it records its action and returns a Rec output.
# 4. g calls backward_pass which returns the gradient df/dx.

# And some background info:
# 5. How recording is done.
# 6. How new primitives and their gradients are defined.
# 7. How higher order gradients work.


# Details:

# 1. g is called with the same inputs as f.
# 1.1 g supports both regular and keyword args.
# 1.2 only one of the regular args is the gradient target, specified by the argnum argument of grad (defaults to 1).
# 1.3 in a typical model f would take parameters in the first arg and return loss.
# 1.4 to support multiple parameters, they can be grouped in a single arg using Array, Dict, or Tuple.

"""

    grad(fun, argnum=1)

Take a function `fun(X...)->Y` and return another function
`gfun(X...)->dXi` which computes its gradient with respect to
positional argument number `argnum`. The function `fun` should be
scalar-valued. The returned function `gfun` takes the same arguments
as `fun`, but returns the gradient instead. The gradient has the same
type and size as the target argument which can be a Number, Array,
Tuple, or Dict.

"""
function grad(fun::Function, argnum::Int=1)
    @dbg ("[grad",fun,argnum)
    function gradfun(args...; kwargs...)
        @dbg ("[gfun",fun,args...,kwargs...)
        @prof "FORW" fp = forward_pass(fun, args, kwargs, argnum) # returns (start, end, tape)
        @prof "BACK" bp = backward_pass(fp...)
        @dbg ("]gfun",fun,args...,kwargs...,'→',bp)
        return bp
    end
    @dbg ("]grad",fun,argnum,'→',gradfun)
    return gradfun
end

"""

    gradloss(fun, argnum=1)

Another version of `grad` where the generated function returns a
(gradient,value) pair.

"""
function gradloss(fun::Function, argnum::Int=1)
    function gradfun(args...; kwargs...)
        fp = forward_pass(fun, args, kwargs, argnum)
        bp = backward_pass(fp...)
        return bp, getval(fp[2])
    end
    return gradfun
end

# 2. g calls forward_pass which boxes argnum'th arg x in a Rec type and calls f(Rec(x))
# 2.1 f must be defined generically to accept Rec arguments.
# 2.2 before the call a new Tape (tape1) is created (this is the only place a new tape is created)
# 2.3 we box the argument if it is not already (as may happen in higher order derivatives) and add it to the new tape
# 2.4 f is called with the boxed argument
# 2.5 the downstream operations on x are recorded on all its tapes
# 2.6 the output of f (end_box) could be a boxed Rec or a regular value (if it does not depend on x)

function forward_pass(fun, args, kwargs, argnum)
    @dbg ("[fwd$argnum", fun, args..., kwargs...)
    tape = Tape()
    arg_wrt = args[argnum]
    if isa(arg_wrt,Rec)
        # @MikeInnes: This fixes a silent bug where
        # grad(x -> x*grad(y -> x+y)(x))(5.0) == 2
        # grad(x -> x*grad(y -> x+y)(1x))(5.0) == 1
        # (the correct answer is 1).
        arg_wrt = identity(arg_wrt)
        Node(arg_wrt, tape) # pushes the existing Rec into new Tape, giving it a new outgrad.
        start_box = arg_wrt
    else
        start_box = Rec(arg_wrt,tape)
    end
    args = Any[args...] # to make args writeable
    args[argnum] = start_box
    @dbg ("[call", fun, args..., kwargs...)
    end_box = fun(args...; kwargs...) # 4458
    @dbg ("]call", fun, args..., kwargs..., '→', end_box)
    @dbg ("]fwd$argnum", '→', start_box, end_box, tape)
    return start_box, end_box, tape
end

# forward_pass type: ((N(X)->N(Y)/Y),N(X)/X,K,I)->(N(X),N(Y)/Y,T)
# forward_pass deps: Tape, Rec, Node


# 3. If a primitive operator inside f gets a Rec input, it records its
# action and returns a Rec output. This is done by dispatching to
# `forw`.

"""
    forw(f, args...; kwargs...)

Primitive operators dispatch `f(args...; kwargs...)` to
`forw(f,args...; kwargs...)` if any of the `args` is a Rec.  `forw`
unboxes the arguments, calls f, boxes and returns the result,
recording the result and its dependencies on each boxed argument.
"""
function forw(f, args...; kwargs...)
    @dbg ("[forw", f, args..., kwargs...)
@prof "forw.$f" begin
    argvals = getval.(args)       # 31
    result = f(argvals...; kwargs...) # 4959
    for argnum = 1:length(args)
        arg = args[argnum]
        isa(arg,Rec) || continue
        for t=1:length(arg.tapes)
            tape = arg.tapes[t]
            iscomplete(tape) && continue
            parent = arg.nodes[t]
            if !isa(result,Rec) 
                result = Rec(result, tape; func=f, args=args, kwargs=kwargs)
                rnode = result.nodes[1]
            else
                s = findeq(result.tapes, tape)
                if s > 0
                    rnode = result.nodes[s]
                else
                    rnode = Node(result, tape)
                end
            end
            rnode.parents[argnum] = parent
        end
    end
    if DEBUGTAPE; _debugtape(result,argvals); end
    @dbg ("]forw", f, args..., kwargs..., '→', result)
end # prof begin
    return result
end # function forw

# forw deps: Rec, Node, iscomplete, findeq

DEBUGTAPE=false
debugtape(b::Bool)=(global DEBUGTAPE=b)
function _debugtape(result, argvals)
    if !isa(result,Rec); return; end
    @assert length(result.tapes) == length(result.nodes) == 1
    tp = result.tapes[1]
    n = result.nodes[1]
    i = findeq(tp,n)
    p = ntuple(length(n.parents)) do j
        if isassigned(n.parents,j)
            findeq(tp,n.parents[j])
        elseif isa(argvals[j],Number) || isa(argvals[j],Symbol) || isa(argvals[j],AbstractRange)
            argvals[j]
        else
            0
        end
    end
    println("$i. $f$p")
end

"""
    getval(x)

Unbox `x` if it is a boxed value (`Rec`), otherwise return `x`.
"""
getval(x) = (if isa(x, Rec); x.value; else; x; end)  # we never create Rec(Rec).

# findfirst uses == which is inefficient for tapes, so we define findeq with ===
function findeq(A,v)
    @inbounds for i=1:length(A)
        if A[i] === v
            return i
        end
    end
    return 0
end


# 4. g calls backward_pass which returns the gradient df/dx.

# 4.1 backward_pass is called with start_box: Rec(x), end_box:
# f(Rec(x)) (which may or may not be a boxed Rec), and the tape
# created by the corresponding forward_pass.  Note that Rec(x) may
# point to more tapes in case of a higher order gradient.  It returns
# the gradient wrt the start_box.

function backward_pass(start_box, end_box, tape)
    @dbg ("[back",start_box,end_box,tape)

# 4.2 If end_box is not a Rec on the given tape, we return zero
# df/fx if x is a bits type, `nothing` otherwise.  end_box may not
# be a Rec if the output of f does not depend on x.

    if !isa(end_box, Rec) || 0==(tapeidx=findeq(end_box.tapes, tape))
        @dbg "Output seems independent of input. Returning zero gradient."
        if isa(start_box,Number); return zero(start_box); else; return nothing; end
    end

    if !isa(end_box.value, Number)
        error("grad requires a scalar-valued function, got $(end_box.value)")
    end

# 4.3 backward_pass resets all node gradients except for the scalar
# output Rec whose gradient is set to 1.0.

    end_box.nodes[tapeidx].outgrad = one(end_box.value)

# We need to complete!(tape) to prevent recording during backward_pass.

    complete!(tape)

# 4.4 the tape is read in reverse and for each node with a non-zero
# outgrad its ingrads are computed using the back methods.

    for n in tape[end-1:-1:1]  # note the end-1 because we pushed an eot marker
        if n.outgrad === nothing; continue; end
        @prof "NODE" r = n.rec
        @dbg ("[node", r.func, r.args..., r.kwargs..., '→', r, '←', n.outgrad)
        for i=1:length(n.parents)
            isassigned(n.parents,i) || continue
            @prof "GRAD" p = n.parents[i]
            #The following was buggy, we need r here not r.value to make higher order gradients work!
            #@prof "back$i.$(r.func)" og = r.func(Grad{i},n.outgrad,r.value,r.args...;r.kwargs...) # 4887
            #@prof "back$i.$(r.func)" og = r.func(Grad{i},n.outgrad,r,r.args...;r.kwargs...)
            @prof "back$i.$(r.func)" og = back(r.func,Val(i),n.outgrad,r,r.args...;r.kwargs...)
            @dbg ("[sum$i",p.rec,'=',p.outgrad,"+",og)
            @prof "sumg$i.$(r.func)" p.outgrad = sum_outgrads(p.outgrad, og) # 1141
            @dbg ("]sum$i",p.rec,'=',p.outgrad)
        end
        @dbg ("]node", r.func, r.args..., r.kwargs..., '→', r, '←', n.outgrad)
    end

# 4.5 tape[1].outgrad is returned.  How do we know this is the
# correct gradient df/dx?  Only x and its descendents are marked as
# Recs and recorded on the tape. In the beginning the only non-empty
# outgrad is the one for the end_box.  Since the end_box is a
# boxed Rec (otherwise we returned 0/nothing), it must depend on
# input x.  The input is the first thing recorded on tape by
# forward_pass, thus will be the last thing whose gradient is seen.
# If there are Recs influenced by x but do not influence the
# end_box, their outgrad will remain empty, thus only the necessary
# gradients are computed.

    @dbg ("]back",start_box,end_box,tape,'→',tape[1].outgrad)
    return tape[1].outgrad
end

# back deps: complete!, sum_outgrads


# 5. How recording is done.

# 5.1 Rec: g=grad(f) calls forward_pass which calls f with one
# argument boxed in a Rec type.  The primitives inside f call their
# forw methods when one of their arguments is a Rec.  The results
# of these forw methods are also boxed in Rec types, which will
# cause downstream primitives to be recorded as well.  The final
# output of f, if not independent of the input, will thus be a Rec.

# 5.2 Node: Each result Rec created by a primitive keeps track of the
# function and the arguments that created the Rec.  Because a Rec may
# need to be recorded in multiple tapes for higher order derivatives
# (see Sec. 7) these dependencies are kept in a separate data
# structure called a Node.  The parents field of a Node is an array
# that points to the Nodes of the arguments.  The gradient wrt the
# result is kept in the outgrad field.  A node can have multiple
# descendents each of which will send a gradient which get accumulated
# in the outgrad field using sum_outgrads.

# 5.3 Tape: When forward_pass is done, we have the computation graph
# (dependency tree) of the result recorded in Nodes.  However we also
# need the time order in which these Nodes were created for the
# backward_pass.  The gradient functions of all the children of a node
# need to be called before its own gradient function.  For example if
# z depends on x and y, and y depends on x, we want to compute the
# gradients in z-y-x order.  If we do it in z-x-y order, the gradient
# function of x will be called before its descendent y.  Thus we keep
# a Tape which is an array of Nodes in the order they were created. A
# Tape represents a computational graph for one forward_pass. Higher
# order gradients will enter forward_pass multiple times creating
# multiple tapes.

mutable struct Node
    rec
    outgrad
    parents::Vector{Node}
    Node(b) = new(b, nothing, Array{Node}(undef,length(b.args)))
end

const Tape = Vector{Node}

mutable struct Rec{T}
    value::T
    func::Function
    args::Tuple
    kwargs # ::Vector: This is no longer a vector in 0.7 
    tapes::Vector{Tape}
    nodes::Vector{Node}
end

function Rec(value, tape::Tape=Tape(); func=rand, args=(), kwargs=())
    self = Rec(value,func,args,kwargs,Tape[tape],Array{Node}(undef,1))
    node = Node(self)
    push!(tape,node)
    self.nodes[1] = node
    return self
end

function Node(b::Rec, t::Tape) # assumes b is not already in t
    n = Node(b)
    push!(t, n)
    push!(b.nodes, n)
    push!(b.tapes, t)
    return n
end

# Primitives with Rec arguments may be called during the
# backward_pass. We do not want those primitives being recorded any
# more (at least on the tape created by the corresponding
# forward_pass, see Sec 7 for details).  We stop the recording on a
# Tape by calling its complete! method.

let eot = Node(Rec(nothing))
    global iscomplete, complete!
    iscomplete(a::Tape)=(!isempty(a) && a[end]===eot)
    complete!(a::Tape)=push!(a,eot)
end


# 6. How new primitives and their gradients are defined.

# 6.1 Primitives

# AutoGrad primitives record their actions when they are called with
# some arguments boxed in Recs (However, see 6.3 for
# undifferentiable primitives). Julia supports multiple dispatch,
# i.e. a single function can have multiple methods with different arg
# types.  AutoGrad supports multiple dispatch for primitives and
# gradients, i.e. only some of the methods of a function can be
# defined as primitives and have gradients.  Calls to a particular
# method where some arguments are boxed in Recs are directed to the
# forw function. The following example makes `sin(x::Number)` a
# primitive, but says nothing about e.g. `sin(x::Array)`.

#     sin(x::Rec{T}) where {T<:Number} = forw(sin,x)

# With multiple arguments, things get a bit more complicated.  There
# is no easy way to say "at least one argument is a Rec" in Julia.
# So one must define methods for all 2^N-1 combinations for
# boxed/unboxed arguments to be safe.  This example makes
# hypot(x1::Real,x2::Real) a primitive:

#     hypot(x1::Rec{T},x2::Rec{S}) where {T<:Real,S<:Real} = forw(hypot,x1,x2)
#     hypot(x1::Rec{T},x2::S)      where {T<:Real,S<:Real} = forw(hypot,x1,x2)
#     hypot(x1::T,x2::Rec{S})      where {T<:Real,S<:Real} = forw(hypot,x1,x2)

# I wrote the @primitive macro in macros.jl to automate this process.
# One restriction is the inability to target parametric methods such
# as `f(AbstractArray{T}) where {T<:Number}`.  Julia did not support
# `f(Rec{A}) where {T<:Number,A<:AbstractArray{T}}` when I wrote this.

# One could also choose to be lazy and just say:

#     hypot(x...) = forw(hypot,x...)

# This would send any argument combination not covered by regular
# hypot methods to the forw function, which presumably includes calls
# with boxed arguments.  This is dangerous for several reasons: (1)
# the Julia base may contain a typeless method (e.g. it does for
# `vcat`) we are overwriting. (2) this catches boxed calls to hypot
# methods we may not support yet.  So generally I would not recommend
# it.

# 6.2 Gradients

# In AutoGrad, gradients are defined using back methods that have the
# following signature:

#     back(f,Val(i),dy,y,x...) => dx[i]

# Here `f` is the name of original function, Val(i) is a constant that
# specifies the gradient wrt the i'th argument, `dy` is the gradient
# wrt the output `y`, and `x...` are the input arguments.  In this
# case `f` was originally called with `f(x...)` and returned `y`.
# Somebody handed us the gradient `dy` wrt the output `y` and
# `back(f,Val(i),...)` above is going to give us the gradient `dx[i]`
# wrt the i'th argument `x[i]`.

# Note that type declarations on the x's can be used to specialize
# this gradient to any method of the function `f`.  Here is the
# gradient for `sin`:

# `back(::typeof(sin),::Val{1}, dy, y, x::Rec{T}) where {T<:Number} = dy*cos(x)

# For the second example a different gradient method is needed for
# each argument:

# `back(::typeof(hypot),::Val{1},dy,y,x1::Rec{T},x2::Rec{S}) where {T<:Real,S<:Real} = (dy*x1/y)`
# `back(::typeof(hypot),::Val{2},dy,y,x1::Rec{T},x2::Rec{S}) where {T<:Real,S<:Real} = (dy*x2/y)`

# And of course we need four more definitions for the other
# boxed/unboxed argument combinations, which the @primitive macro
# generates automatically.

# Finally, there are three cases of zero gradients that need to be
# handled:

# 6.3 Undifferentiable functions

# Piecewise constant functions such as `sign`, and non-numeric
# functions such as `size` are not differentiable wrt any of their
# arguments.  Unlike primitives, these functions do not need to record
# their action or return a boxed Rec.  They can just unbox their
# arguments and return an unboxed value:

# `size(a::Rec,i...)=size(a.value,i...)`

# The @zerograd macro defined in macros.jl can be used to automate this.

# 6.4 Undifferentiable wrt unboxed arguments

# Methods such as `sum(a::Array,i::Int)` are only differentiable wrt
# some of their arguments (here `a` but not `i`).  These methods must
# record when their differentiable argument(s) are boxed and return
# boxed values.  If we are certain that their undifferentiable
# arguments are never going to be boxed, we can leave their gradients
# undefined:

#     sum(a::Rec{T},i::Int) where {T<:Array} = forw(sum,a,i)
#     back(::typeof(sum),::Val{1},dy,y,a::Rec{T},i::Int) where {T<:Array} = dy.+zero(a)

# (Note: the calling convention of sum was changed to `sum(a;dims=i)`
# which makes this example outdated, but the concept stands.)

# 6.5 Undifferentiable wrt boxed arguments

# Finally, in the rare cases when an undifferentiable argument can be
# boxed, its gradient must be defined and must return `nothing`.  The
# utility function `ungetindex` in getindex.jl which uses its first
# argument's shape as a template is one example of this rare class.

# 6.6 sum_outgrads

sum_outgrads(a::Number, b::Number)=a+b
sum_outgrads(a::Tuple, b::Tuple)=tuple([sum_outgrads(x,y) for (x,y) in zip(a,b)]...)
sum_outgrads(a::AbstractDict, b::AbstractDict) = (z=similar(a); for d in (a,b), (k,v) in d; z[k]=sum_outgrads(v,get(z,k,nothing)); end; z)
# We could have Array{Array} and Array{Any} added:
sum_outgrads(a::AbstractArray{T},b::AbstractArray) where T = (if isbitstype(T); (a+b); else; T[sum_outgrads(x,y) for (x,y) in zip(a,b)]; end)
# sum_outgrads needs to be a primitive for higher order gradients:
sum_outgrads(a::Rec,b::Rec)=forw(sum_outgrads,a,b)
sum_outgrads(a::Rec,b)=forw(sum_outgrads,a,b)
sum_outgrads(a,b::Rec)=forw(sum_outgrads,a,b)
back(::typeof(sum_outgrads),::Val{N},dy,y,x1,x2) where N = dy
# we use `nothing` to indicate zero gradients
sum_outgrads(::Nothing,::Nothing)=nothing
sum_outgrads(a::Rec,::Nothing)=a   # to avoid ambiguity
sum_outgrads(::Nothing,a::Rec)=a   # to avoid ambiguity
sum_outgrads(a,::Nothing)=a
sum_outgrads(::Nothing,a)=a


# 7. How higher order gradients work.

# Say g=grad(f) and h=grad(g) and we call h(x).
# h(x) calls forward_pass(g,x)
# merge_tapes in forward_pass(g,x) is a noop because x is not a Rec.
# forward_pass(g,x) wraps x in v1=Rec(x,t1:n1) with tape t1 and node n1 and calls g(v1).
# g(v1) calls forward_pass(f,v1), which creates v2=Rec(x,t2:n2)
# merge_tapes in forward_pass(f,v1) creates v3=Rec(x,[t1:n31,t2:n32]) with parents n31->n1, n32->n2.
# forward_pass(f,v1) calls f(v3)
# primitives in f(v3) push their result Nodes on both [t1,t2] and record parents on each tape separately.
# f(v3) returns v4=Rec(y,[t1:n41,t2:n42]).
# g(v1) calls backward_pass(f)(v2,v4,t2).
# backward_pass(f) calls complete!(t2) and starts processing the nodes on t2 in reverse.
# the nodes on t2 only point to other nodes in t2, so backward_pass(f) fills outgrads on t2.
# backward_pass(f) calls gradient methods of the recorded primitives in f(v3).
# the operations of gradient methods are recorded only on t1, that is why we need iscomplete(t2) once we start backward_pass on t2.
# backward_pass(f) returns v5=Rec(df/dx,t1:n5) which becomes the output of forward_pass(g,x)
# h(x) calls backward_pass(g)(v1,v5,t1).
# backward_pass(g) calls gradient methods recorded in t1.
# even though some inputs are Recs again, nothing gets recorded and all primitives return values because t1 is complete.
# backward_pass(g) returns a regular value which becomes the output of h(x).


