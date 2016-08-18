# Contents:

# Here are the rough steps performed by g(x) where g=grad(f):
# 1. g is called with the same inputs as f.
# 2. g calls forward_pass which boxes x in a Node type and calls f(Node(x)).
# 3. If a primitive operator inside f gets a Node input, it records its action and returns a Node output.
# 4. g calls backward_pass which returns the gradient df/dx.

# And some background info:
# 5. How recording is done.
# 6. How new primitives and their gradients are defined.
# 7. How higher order gradients work.
# 8. Support functions.

# Details:

# 1. g is called with the same inputs as f.
# 1.1 g supports both regular and keyword args.
# 1.2 only one of the regular args is the gradient target, specified by the argnum argument of grad.
# 1.3 in a typical model f would take parameters, return loss, with data kept in global variables.
# 1.4 to support multiple parameters, they can be grouped in a single arg using Array, Dict, or Tuple.

"""
grad(fun, argnum=1) -> gradfun    

* fun: X->Y    
* gradfun: X->dX   

Returns a function which computes the gradient of `fun` with respect to
positional argument number `argnum`. The returned function takes the same
arguments as `fun`, but returns the gradient instead. The function `fun`
should be scalar-valued. The gradient has the same type as the argument.
"""
function grad(fun::Function, argnum::Int=1)
    function gradfun(args...; kwargs...)
        backward_pass(forward_pass(fun, args, kwargs, argnum)...)
    end
    @dbgcore((:grad,name(gradfun,(Symbol("D$argnum"),name(fun)))))
    return gradfun
end


# 2. g calls forward_pass which boxes x in a Node type and calls f(Node(x))
# 2.1 f must be defined generically to accept Node arguments.
# 2.2 before the call a CalculationTape (tape1) is created (this is the only place a new tape is created)
# 2.3 n1=Node(x,tape1) is created, each node is associated with one or more tapes.
# 2.4 x may already be a n0=Node(x,tape0) from another tape (this happens with higher order derivatives)
# 2.5 if 2.4, another n2=Node(x,[tape0,tape1]) is created by merge_tapes on both tapes with dependencies on n0 and n1.
# 2.6 if not 2.4, merge_tapes simply returns n1.
# 2.7 f is called with n1 or n2 (depending on 2.4).
# 2.8 if n2, the downstream operations on x are recorded on both tape0 and tape1.
# 2.9 the output of f (end_node) could be a Node or a regular value (if it does not depend on x)

"""
forward_pass(fun, args, kwargs, argnum) -> (start_node, end_node, tape)

Wraps the `argnum`'th arg in a Node with an empty tape and calls `fun`
which returns a Node or a regular value.  The input node, the output
and the tape are returned.  Note that forward_pass is only called for
the top level function, the internal operations use regular `call`.
This is the only place where a tape is created.  Multiple tapes only
enter the picture for higher order derivatives.
"""    
function forward_pass(fun, args, kwargs, argnum)
    @dbgcore((:forw, argnum, name(fun), map(name,args)..., map(name,kwargs)...))
    tape = CalculationTape()
    arg_wrt = args[argnum]
    start_node = Node(float(getval(arg_wrt)), Any[tape])
    args = Any[args...] # to make args writeable
    args[argnum] = merge_tapes(start_node, arg_wrt)
    @dbgcore((:fcall, name(fun), map(name,args)..., map(name,kwargs)...))
    end_node = fun(args...; kwargs...)
    return start_node, end_node, tape
end

# forward_pass: ((N(X)->N(Y)/Y),N(X)/X,K,I)->(N(X),N(Y)/Y,T)
# deps: CalculationTape, Node, getval, merge_tapes
# also float extended to handle cell arrays, Tuple and Dict in util.jl
getval(x) = isa(x, Node) ? x.value : x  # we never create Node(Node).

# 3. If a primitive operator inside f gets a Node input, it records its action and returns a Node output.

# 3.1 We implement this by dispatching f to r=recorder(f) if any of
# its arguments is a Node.  r unboxes the arguments, calls f, boxes
# and returns the result, recording the result and its gradient
# functions for each argument.

"""
recorder(fun) returns rfun, a recording version of fun.  rfun is defined with
a generic signature r(args...; kwargs...) and is intended to catch all
invocations that have at least one Node argument.
"""
function recorder(f)
    #@dbgcore((:recorder,f))
    function r(args...; kwargs...)
        @dbgcore((:call, name(f), map(name,args)..., map(name,kwargs)...))
        argvals = Any[args...]
        ops = []
        tapes = []
        found_node = false

# 3.2 r goes through the arguments and unboxes any that are Nodes.
# For each unboxed Node, its tape, argnum, and ReverseNode is stored.

        for i=1:length(args)
            arg = args[i]
            if isa(arg, Node)
                found_node = true
                argvals[i] = arg.value
                # if i in p.zero_grads; continue; end               # we represent zero_grads using gradmakers that return 0 instead of a function
                for (tape, parent_rnode) in arg.tapes               # Node.tapes is a Dict{Tape,ReverseNode}
                    if !iscomplete(tape)                            # why do we need iscomplete? to prevent recording during the backward_pass unless we are doing higher order derivatives.
                        push!(ops, (tape, i, parent_rnode))         # ops should be called args or inputs!
                        push!(tapes,tape)                           # duplicates will be handled by Node constructor
                    end
                end
            end
        end

# 3.3 If no Nodes found we throw an error (the recorder method was
# supposed to catch calls with Node arguments).

        found_node || throw(MethodError(f, argvals))            # Otherwise undefined methods lead to infinite loop

# 3.4 The primitive is called with unboxed arguments.
        @dbgcore( (:rcall,name(f),map(name,argvals)...,map(name,kwargs)...))
        result = f(argvals...; kwargs...)

# 3.5 ops can be empty if no Nodes, zero_grads, or iscomplete(tape).
# No Nodes case is impossible, we throw an error.
# zero_grads is handled differently.
# iscomplete is needed to prevent recording during backward_pass unless higher order derivatives.

        if !isempty(ops) 

# 3.6 We box the result in a Node attached to all the tapes we
# encountered in 3.2.  This boxed result gets returned.

            result = Node(result, tapes) # TODO: make sure it is not a problem to have extra tapes because we check for zero_grads below.

# 3.7 For each of our Node inputs, we create a gradient function for
# the specific argnum of the primitive.  This gradfun takes dy and
# returns dx and has access to the original x,y through a closure.
# The gradfun and the ReverseNode of the associated input is stored in
# parent_grad_ops.

            for (tape, argnum, parent) in ops                       
                @dbgcore((:gcall,name(f),argnum,name(result),map(name,args)...,map(name,kwargs)...))
                gradfun = f(Grad{argnum}, result, args...; kwargs...) # Creates a node specific gradfun (dy->dx) with x,y in a closure
                gradfun == 0 && (warn("gradfun=0"); continue) # indicates zero_grad arguments
                @dbgcore(name(gradfun,(Symbol("D$argnum"),f,:out,name(result),:args,map(name,args)...,map(name,kwargs)...))) # Record for debugging
                rnode = result.tapes[tape]
                push!(rnode.parent_grad_ops, (gradfun, parent))
                @dbgcore((:deps,name(tape),rnode))
            end
        end
        return result
    end
    return r
end


# 4. g calls backward_pass which returns the gradient df/dx.

# 4.1 backward_pass is called with start_node: Node(x), end_node:
# f(Node(x)), which may or may not be a Node, and the tape created by
# the corresponding forward_pass.  Note that Node(x) may point to more
# tapes in case of a higher order gradient.

"""
backward_pass(start_node, end_node, tape) -> gradient wrt start_node.value
"""
function backward_pass(start_node, end_node, tape)
    @dbgcore((:back,name(start_node),name(end_node),name(tape),map(name,tape)...))

# 4.2 If end_node is not a Node on the given tape, we return zero df/fx.
# end_node may not be a Node if the output of f does not depend on x.
# Q: Could the end_node be a Node but in a different tape?    
# A: Could df/dx be a Node?  Yes, for a higher order gradient.
# A: Should zeros_like return a Node if the input is a Node?  No need, it is a constant.

    if !isa(end_node, Node) || !haskey(end_node.tapes, tape)    # This may happen e.g. if the function returns a constant
        @dbgcore( "Output seems independent of input. Returning zero gradient.")
        return zeros_like(start_node)
    end
    if !isa(getval(end_node), Number)
        error("grad requires a scalar-valued function, got $(getval(end_node))")
    end

# 4.3 backward_pass resets all node gradients except for the scalar
# output Node whose gradient is set to 1.0.
# A: Why do we need complete!(tape)?  To prevent recording during backward_pass.

    for node in tape                                            # tape is created by forw_pass
        node.outgrads = []
    end
    end_node.tapes[tape].outgrads = [1.0]                       # end_node.tapes[tape] is the ReverseNode corresponding to end_node::Node

    complete!(tape)

# 4.4 the tape is read in reverse and for each node with a non-empty
# outgrad its ingrads are computed using the closures recorded by the
# primitive recorders.

    cur_outgrad = nothing
    for node in tape[end-1:-1:1]                                # note the end-1 because we pushed a marker to complete
        if !isempty(node.outgrads)
            @dbgcore((:sum1,name(node),:args,map(name,node.outgrads)...))
            cur_outgrad = sum_outgrads(node.outgrads...)
            # This bombs when we have different types of Dict or Array
            # typeof(getval(cur_outgrad)) == typeof(node.node.value) || error("Type mismatch: y=$(node.node.value) dy=$(getval(cur_outgrad))")
            @dbgcore((:sum2,name(node),:out,name(cur_outgrad)))
            for (gradfun, parent) in node.parent_grad_ops
                @dbgcore((:back1,name(cur_outgrad),name(gradfun)))
                og = gradfun(cur_outgrad)
                push!(parent.outgrads, og)
                @dbgcore((:back2,name(og),name(gradfun)))
            end
        end
    end

# 4.5 the last outgrad is returned.  How do we know this is the
# correct gradient df/dx?  Only x and its descendents are marked as
# Nodes and recorded on the tape. In the beginning the only non-empty
# outgrad is the one for the end_node.  Since the end_node is a Node
# (otherwise we return 0), it must depend on input x.  The input is
# the first thing recorded on tape by forward_pass, thus will be the
# last thing whose gradient is seen.  If there are Nodes influenced by
# x but do not influence the end_node, their outgrads will remain
# empty, thus only the necessary gradients are computed.  However, if
# there is a nondifferentiable operation the chain breaks!  So let's
# test for it anyway.

    isempty(tape[1].outgrads) && error("Output independent of input?")
    return cur_outgrad
end


# 5. How recording is done.

# 5.1 Node: g=grad(f) calls forward_pass which calls f with one
# argument boxed in a Node type.  The primitives inside f call their
# recorder methods when one of their arguments is a Node.  The results
# of these recorder methods are also boxed in Node types, which will
# cause downstream primitives to be recorded as well.  The final
# output of f, if not independent of the input, will thus be a Node.

if !isdefined(:Node)            # to prevent error on reload
"""
Node(value, tapes) creates a new Node:

1. in forward_pass for the argument we are taking gradient w.r.t.
2. for the output of a primitive operation with Node input.

When a Node is created, it pushes corresponding ReverseNodes on each
of the tapes given in the second argument, and records pointers to
each tape and its ReverseNode in its `tapes` dictionary.  These
ReverseNodes have empty parent_grad_ops and outgrads which are written
by call and back respectively.  Ordinarily there is only one tape
(unless we do higher order derivatives).

"""
type Node{T}; value::T; tapes::ObjectIdDict; end
end #if !isdefined(:Node)

function Node(value, tapes=Any[CalculationTape()])     # arg tapes is an Array
    self = Node(value, ObjectIdDict())                      # field tapes is an ObjectIdDict(CalculationTape=>ReverseNode)
    for tape in tapes
        haskey(self.tapes, tape) && continue
        new_rnode = ReverseNode(self)
        push!(tape, new_rnode)                              # This is the only place new elements are added to a tape.
        self.tapes[tape] = new_rnode
    end
    @dbgcore((:node,self))
    return self
end

if !isdefined(:Nval)
"""
Nval stands for Node or Value.  Convenience type useful when
defining gradmakers f(::D{N},y,x...).  We need to be able to define
gradmakers with different type signatures.  y is always a Node, and
at least one of the x's is a Node, but other x's may or may not be
Nodes.  So it is useful to have a type that represents that:
"""
typealias Nval{T} Union{T,Node{T}}
end

# 5.2 ReverseNode: Each result Node created by a primitive keeps track
# of the argument Nodes of that primitive (the non-Node arguments need
# not be recorded since they do not depend on the input of f).  Along
# with each argument Node i, a gradient function is recorded that will
# turn the gradient wrt the result into a gradient wrt argument i.
# For reasons that will become clear, these dependencies are kept in a
# separate data structure called a ReverseNode in its parent_grad_ops
# field).  The gradient wrt the result is kept in the outgrads field.
# parent_grad_ops is an array because a node can have multiple
# arguments.  outgrads is an array because a node can have multiple
# descendents each of which will push a gradient to outgrads to be
# summed.

"""
ReverseNode is a plain type with three slots:

* `parent_grad_ops`: `call` fills this array with (gradfun,parent_rnode) pairs for each Node argument.
* `outgrads`: used by backward_pass, array of gradients for this node (has multiple elements if fanout > 1).
* `node`: corresponding Node
"""    
type ReverseNode; node; parent_grad_ops; outgrads; end
ReverseNode(node) = ReverseNode(node, [], [])


# 5.3 CalculationTape: When forward_pass is done, we have the
# computation graph (dependency tree) of the result recorded in
# ReverseNodes.  However we also need the time order in which these
# ReverseNodes were created for the backward_pass.  The gradient
# functions of all the children of a node need to be called before its
# own gradient function.  For example if z depends on x and y, and y
# depends on x, we want to compute the gradients in z-y-x order.  If
# we do it in z-x-y order, the gradient function of x will be called
# before its descendent y.  Thus we keep a CalculationTape which is an
# array of ReverseNodes in the order they are created.

# Primitives with Node arguments may be called during the
# backward_pass. We do not want those primitives being recorded any
# more (at least on the tape created by the corresponding
# forward_pass, see Sec 7 for details).  We stop the recording on a
# CalculationTape by calling its complete! method.

# A Node may have multiple corresponding ReverseNodes in multiple
# CalculationTapes.  It keeps track of all its ReverseNodes via the
# tapes field which is a CalculationTape=>ReverseNode dictionary.
# Multiple tapes are only needed for higher order gradients, see
# Section 7, How higher order gradients work.

"CalculationTape is an array of ReverseNodes that supports `complete!` and `iscomplete`."
typealias CalculationTape Array{ReverseNode,1}
iscomplete(a::CalculationTape)=(!isempty(a) && a[end].node==nothing)
complete!(a::CalculationTape)=push!(a,ReverseNode(nothing))


# 6. How new primitives and their gradients are defined.

# 6.1 @primitive

"""
`@primitive f(args...; kwargs...)` causes f to call its recorder
method for the argument signature provided (see `recorder`).  Note
that the recorder method will give an error unless one of the
arguments is a Node. Examples:

`@primitive log(x...; o...)` will cause all calls to `log` not matched
by any other method to call the recorder method.  This is not
recommended, it is usually better to specify argument types.

`@primitive log` is defined as syntactic sugar for `@primitive log(x...; o...)`.

`@primitive getindex(x::Node, i)` will cause `getindex` to call its
recorder method only if the first argument is a Node.

`@primitive sum{T<:Number}(a::Node{Array{T}})` will cause `sum` to
call its recorder method for Nodes that box Arrays of Number subtypes.
"""
macro primitive(fx)
    isa(fx, Symbol) && (fx = :($fx(x...;o...)))
    (isa(fx, Expr) && fx.head == :call) || error("Malformed @primitive $fx, see `doc @primitive`.")
    rx = notypes(fx)
    f = rx.args[1]
    rx.args[1] = r = gensym()
    esc(:(local $r = recorder($f); $fx=$rx))
end

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

# 6.2 @zerograd

"""
`@zerograd f(args...; kwargs...)` allows f to handle its Node inputs
by unboxing them like @primitive, but unlike @primitive it does not
record its actions or return a Node result.  Some functions, like
sign(), have zero gradient.  These need to handle Node inputs, but do
not need to record anything and can return regular values.  Their
output can be treated like a constant in the program.  Use the
@zerograd macro for those.  Note that kwargs are NOT unboxed. (other
exceptions to recording: gradient functions, some utilities, zerograd
functions, a completed tape).
"""
macro zerograd(fx)
    isa(fx, Symbol) && (fx = :($fx(x...;o...)))
    (isa(fx, Expr) && fx.head == :call) || error("Malformed @zerograd $fx, see `doc @zerograd`.")
    rx = notypes(fx)
    f = rx.args[1]
    rx.args[1] = r = gensym()
    esc(:(local $r = unboxnodes($f); $fx=$rx))
end

function unboxnodes(f)
    u(x...; o...)=f(map(getval,x)...; o...)
    return u
end


# Finally, some functions may have non-zero gradients for some
# arguments, zero for others.  My untested method (TODO: test): use
# @primitive, when defining gradients, define the non-zero ones
# normally with f(::Di,y,x...)=(dy->...), and mark the zero gradients
# with gradmaker returning 0 instead of a function: f(::Di,y,x...)=0.

# 6.3 Gradients: For gradients we define a gradmaker for each
# primitive method p and argnum.  The gradmaker returns a gradient
# function (df/dy->df/dx) that has access to the original input/output
# through a closure.  Julia has multiple-dispatch, which means each
# argument type combination for a function might end up calling a
# different method, each potentially requiring different gradients.
# So we store gradmakers in methods called with `f(Grad{N}, y, x...)`.
# `Grad{N}` represents the gradient wrt the N'th argument, y is the
# output and x... are the inputs of the original function.  This way
# we can use method dispatch to find the appropriate gradient by
# specifying types for x.  Example:
# `sin{T<:Number}(::Type{Grad{1}},y::Node{T},x::Node{T})=(dy->dy*cos(x))`

# It gets tiresome to write `Type{Grad{1}}` after a while, here are
# some convenient aliases:

if !isdefined(:Grad)
    immutable Grad{N}; end          # Gradient wrt N'th argument
end
typealias D1 Type{Grad{1}}
typealias D2 Type{Grad{2}}
if !isdefined(:Dn)
typealias Dn{N} Type{Grad{N}}
end

# Some functions do not have gradients wrt some arguments.  Example:
# getindex(array, index) is not differentiable wrt index.  We indicate
# this using a gradmaker function that returns 0 (serving the same
# role as zero_grads in Python autograd).

# 7. How higher order gradients work.

# Say g=grad(f) and h=grad(g) and we call h(x).
# h(x) calls forward_pass(g,x), which wraps x in n1=Node(x,t1:r1) and calls g(n1).
# merge_tapes in forward_pass(g,x) is a noop because x is not a Node.
# g(n1) calls forward_pass(f,n1), which creates n2=Node(x,t2:r2)
# merge_tapes in forward_pass(f,n1) creates n3=Node(x,[t1:r31,t2:r32]) with pointers r31->r1, r32->r2.
# forward_pass(f,n1) calls f(n3)
# ops in f(n3) push their results on [t1,t2] and record gradfuns and parents on each tape separately.
# f(n3) returns n4=Node(y,[t1:r41,t2:r42]).
# g(n1) calls backward_pass(f)(n2,n4,t2).
# backward_pass(f) calls complete!(t2) and starts processing the rnodes on t2 in reverse.
# the rnodes on t2 only point to other rnodes in t2, so backward_pass(f) fills outgrads on t2.
# backward_pass(f) calls gradfuns recorded by ops in f(n3).
# these gradfuns are defined generically like f, they can handle regular or Node input.
# a gradfun is a closure (dy->dx) with an environment (x,y).  dy may be a Node or value, (x,y) are typically Nodes recorded during the call.
# the operations of gradfuns are recorded only on t1, that is why we need iscomplete(t2) once we start backward_pass on t2.
# backward_pass(f) returns n5=Node(df/dx,t1:r5) which becomes the output of forward_pass(g,x)
# h(x) calls backward_pass(g)(n1,n5,t1).
# backward_pass(g) calls gradfuns recorded in t1.
# even though some inputs are Nodes again, nothing gets recorded and all primitives return values because t1 is complete.
# backward_pass(g) returns a regular value which becomes the output of h(x).


# 8. Support functions.

# These are helper functions used in forward_pass, backward_pass, and
# recorder.  We need to be careful about whether they take or return
# Nodes and record their gradients.  merge_tapes, sum_outgrads are defined as primitives.

# 8.1 Node, getval: these box, unbox, and float values.

# 8.2 merge_tapes: Used by forward_pass to support higher order
# gradients. Records its operation if both its arguments are Nodes.
# Its first argument is a newly created Node.  Its second argument is
# the original input argument, which is only a Node if this is a
# higher order gradient, in which case it is a Node that belongs to
# another tape and merge_tapes in effect creates a third Node on both
# tapes that point to their respective parent Nodes.  If the second
# argument is not a Node, simply returns its first arg without
# recording anything.

merge_tapes(x,y) = x
# The derivatives simply pass the gradients back.
merge_tapes(::D1,c,a,b) = (x->x)   
merge_tapes(::D2,c,a,b) = (x->x)
@primitive merge_tapes(x::Node,y::Node)


# 8.3 gradient makers like merge_tapes(::D1,c,a,b) and gradfuns they return

# The gradmakers don't record even when their input is a Node, is that
# a problem?  Let's think about this.  gradmakers are only run during
# forward call of primitives to generate a gradfun.  They are always
# run with Node arguments.  However their output is a Function,
# i.e. does not have a gradient, so they do not need to be recorded
# even though they have Node arguments.  A gradfun is run by back to
# compute dx from dy.  It is a composite function with a single
# argument dy and closure variables (x,y).  These may be Nodes, in
# which case boxing and unboxing will be performed by the primitives
# used in gradfun.

# The gradfun operations will only be recorded for high order
# gradients where the higher order tape is not closed.  If the tapes
# are closed (in the final backward_pass), no recording will be done
# and a regular value will be returned even if some of the inputs are
# Nodes.


# 8.4 zeros_like: creates a structure similar to x but filled with
# zeros.  Used by backward_pass and getindex.

# A: Should it return a Node when its input is a Node?

# In backward_pass only used to give the return value when end_node is
# not a Node on the given tape.  This means the output of f is
# independent of input.  In a simple gradient, backward_pass always
# returns a value.  In a high-order gradient, backward_pass(f) returns
# a Node, and backward_pass(g) returns a value.  However even in
# backward_pass(f) we do not need to return a Node if the value we are
# returning is a constant that does not depend on the input!  So
# zeros_like can always return a regular value and does not need to be
# a primitive.

# In getindex(::D1) zeros_like is never called with a Node input.  So
# it is safe to unbox the input of zeros_like and not box its result.

"""
zeros_like(x) -> value or object similar to x filled with zeros.
Can handle bits types, Array, Tuple, Associative, and Node.
Implementation similar to deepcopy.
TODO: avoid allocating large arrays using `nothing` like Knet.
"""
zeros_like(x) = fill_similar(x,0)
fill_similar(x,v) = fill_internal(x,v,ObjectIdDict())
fill_internal(x::Node,v,d::ObjectIdDict)=fill_check(x.value,v,d)
fill_internal(x::Tuple,v,d::ObjectIdDict)=ntuple(i->fill_check(x[i],v,d), length(x))
fill_internal(x::Associative,v,d::ObjectIdDict)=
    (a=similar(x); for (key,val) in x; a[key]=fill_check(val,v,d); end; a)
fill_internal{T}(x::AbstractArray{T},v,d::ObjectIdDict)=
    (isbits(T) ? fill!(similar(x),T(v)) : T[fill_check(e,v,d) for e in x])
fill_internal{T}(x::T,v,d::ObjectIdDict)=(isbits(T) ? T(v) : error("fill_similar cannot handle $T"))
fill_check(x,v,d::ObjectIdDict)=(haskey(d,x) ? d[x] : d[x]=fill_internal(x,v,d))


# 8.5 sum_outgrads: only used in backward_pass to produce the input to
# a gradient function df/dy.  Does it ever need to record its
# operation and give a Node output?  Gradient functions are closures
# that take df/dy, return df/dx and have the environment (x,y) which
# is the original input/output and almost certainly Nodes.  The input
# df/dy may or may not be a Node.  If we encounter a Node with an open
# tape, we need to record the sum operation.  This will happen in
# higher order derivatives.

# TODO: Instead of maintaining an array of outgrads then summing them, why not keep a sum to avoid allocation?
# (instead of pushing to parent.outgrads, we'd have to call sum directly, deal with Nodes etc.)
# A: what if first outgrad is a Node and others aren't, or vice versa?  handled by @primitive.
# A: what should the output be if some of the inputs are Nodes?  Is sum_outgrads a primitive? Yes.
# A: for array and dict can we just modify the first element of outgrads?  This may be dangerous because the same outgrad may be passed back to more than one target e.g. by `+`.

sum_outgrads(x)=x
sum_outgrads(a::Number, b::Number, c::Number...)=sum([a,b,c...])
sum_outgrads(a::Tuple, b::Tuple, c::Tuple...)=tuple([sum_outgrads(e...) for e in zip(a,b,c...)]...)
sum_outgrads{T}(a::AbstractArray{T},b::AbstractArray{T},c::AbstractArray{T}...) =
    (isbits(T) ? broadcast(+,a,b,c...) : [sum_outgrads(e...) for e in zip(a,b,c...)])
sum_outgrads(a::Associative, b::Associative, c::Associative...) =
    (z=similar(a); for d in (a,b,c...), (k,v) in d; z[k]=v+get(z,k,0); end; z)
sum_outgrads{N}(::Dn{N}, y, x...)=(dy->dy)
@primitive sum_outgrads

# Pretty print for debugging:
_name=ObjectIdDict()
name(f,n)=(_name[f]=n)
name(f)=get(_name,f,f)
name(x::ReverseNode)=Symbol("R$(href(x))")
name(x::Node)=Symbol("N$(href(x))")
name(x::Array)=Symbol("A$(join([href(Ref(x)),size(x)...],'x'))")
name(x::Tuple)=map(name,x)
href(x)=Int(hash(x)%100)

Base.show(io::IO, n::Node) = print(io,"$(name(n))$((name(n.value),[(name(t),name(r)) for (t,r) in n.tapes]...))")
Base.show(io::IO, n::ReverseNode) = print(io,"$(name(n))$((name(n.node.value),map(name,n.outgrads),[(name(y),name(x)) for (x,y) in n.parent_grad_ops]...))")
