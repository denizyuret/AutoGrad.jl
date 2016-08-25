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
    @dbgcore((:grad,fun,argnum))
    function gradfun(args...; kwargs...)
        backward_pass(forward_pass(fun, args, kwargs, argnum)...)
    end
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
    @dbgcore((:forw, argnum, fun, args..., kwargs...))
    tape = CalculationTape()
    arg_wrt = args[argnum]
    start_node = Node(tofloat(getval(arg_wrt)), Any[tape])
    args = Any[args...] # to make args writeable
    args[argnum] = merge_tapes(start_node, arg_wrt)
    @dbgcore((:fcall, fun, args..., kwargs...))
    end_node = fun(args...; kwargs...)
    return start_node, end_node, tape
end

# forward_pass: ((N(X)->N(Y)/Y),N(X)/X,K,I)->(N(X),N(Y)/Y,T)
# forward deps: CalculationTape, Node, getval, tofloat, merge_tapes

# 3. If a primitive operator inside f gets a Node input, it records its action and returns a Node output.

# 3.1 We implement this by dispatching f to r=recorder(f) if any of
# its arguments is a Node.  r unboxes the arguments, calls f, boxes
# and returns the result, recording the result and its gradient
# functions for each argument.  Here is where r gets created:

function recordfn(f)
    function r(args...; kwargs...)
        @dbgcore((:call, f, args..., kwargs...))
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
        @dbgcore((:rcall,f,argvals...,kwargs...))
        result = f(argvals...; kwargs...)

# 3.5 ops can be empty if no Nodes, zero_grads, or iscomplete(tape).
# No Nodes case is impossible, we throw an error.
# zero_grads is handled differently.
# iscomplete is needed to prevent recording during backward_pass unless higher order derivatives.

        if !isempty(ops) 

# 3.6 We box the result in a Node attached to all the tapes we
# encountered in 3.2.  This boxed result gets returned.

            result = Node(result, tapes)

# 3.7 For each of our Node inputs, we create a gradient function for
# the specific argnum of the primitive.  This gradfun takes dy and
# returns dx and has access to the original x,y through a closure.
# The gradfun and the ReverseNode of the associated input is stored in
# parent_grad_ops.

            for (tape, argnum, parent) in ops                       
                @dbgcore((:gcall,f,argnum,:out,result,:args,args...,kwargs...))
                gradfun = f(Grad{argnum}, result, args...; kwargs...) # Creates a node specific gradfun (dy->dx) with x,y in a closure
                if gradfun == 0 # indicates zero_grad arguments
                    @dbgcore("WARNING: gradfun=0 for $f,$argnum")
                    continue
                end
                rnode = result.tapes[tape]
                push!(rnode.parent_grad_ops, (gradfun, parent, f))
                @dbgcore((:deps,tape,rnode))
            end
        end
        return result
    end
    return r
end

# 3.8 We only need one recorder per function, but recorder(f) may be
# called many times for different methods.  To avoid duplication we
# hold recorders in a hash.

make_recorder(d::ObjectIdDict)=(f->get!(d,f,recordfn(f)))

"""
recorder(fun) returns rfun, a recording version of fun.  rfun is defined with
a generic signature r(args...; kwargs...) and is intended to catch all
invocations that have at least one Node argument.
"""
recorder = make_recorder(ObjectIdDict())

# recorder deps: iscomplete, Node, Grad

# 4. g calls backward_pass which returns the gradient df/dx.

# 4.1 backward_pass is called with start_node: Node(x), end_node:
# f(Node(x)), which may or may not be a Node, and the tape created by
# the corresponding forward_pass.  Note that Node(x) may point to more
# tapes in case of a higher order gradient.

"""
backward_pass(start_node, end_node, tape) -> gradient wrt start_node.value
"""
function backward_pass(start_node, end_node, tape)
    @dbgcore((:back,:start,start_node,:end,end_node,:tape,tape,tape...))

# 4.2 If end_node is not a Node on the given tape, we return zero df/fx.
# end_node may not be a Node if the output of f does not depend on x.
# Q: Could the end_node be a Node but in a different tape?    
# A: Could df/dx be a Node?  Yes, for a higher order gradient.
# A: Should zeros_like return a Node if the input is a Node?  No need, it is a constant.

    if !isa(end_node, Node) || !haskey(end_node.tapes, tape)    # This may happen e.g. if the function returns a constant
        @dbgcore("Output seems independent of input. Returning zero gradient.")
        return (isa(start_node,Number) ? zero(start_node) : nothing) # Using nothing for zero arrays and other structures.
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
            @dbgcore((:sum1,node,:args,node.outgrads...))
            cur_outgrad = sum_outgrads(node.outgrads)
            # This bombs when we have different types of Dict or Array
            # typeof(getval(cur_outgrad)) == typeof(node.node.value) || error("Type mismatch: y=$(node.node.value) dy=$(getval(cur_outgrad))")
            @dbgcore((:sum2,node,:out,cur_outgrad))
            for (gradfun, parent, f) in node.parent_grad_ops
                @dbgcore((:back1,cur_outgrad,f))
                og = gradfun(cur_outgrad)
                push!(parent.outgrads, og)
                @dbgcore((:back2,og,f))
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

# back deps: getval, complete!, sum_outgrads

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

# 6.1 Primitives

# AutoGrad primitives record their actions when they are called with
# some arguments boxed in Nodes.  Julia supports multiple dispatch,
# i.e. a single function can have multiple methods with different arg
# types.  AutoGrad supports multiple dispatch for primitives and
# gradients, i.e. only some of the methods of a function can be
# defined as primitives and have gradients.  Calls to a particular
# method where some arguments are boxed in Nodes are directed to the
# recorder function. The following example makes `sin(x::Number)` a
# primitive, but says nothing about e.g. `sin(x::Array)`.

#     local sin_r = recorder(sin)
#     sin{T<:Number}(x::Node{T}) = sin_r(x)

# With multiple arguments, things get a bit more complicated.  There
# is no easy way to say "at least one argument is a Node" in Julia.
# So one must define methods for all 2^N-1 combinations for
# boxed/unboxed arguments.  This example makes
# hypot(x1::Array,x2::Array) a primitive:

#     local hypot_r = recorder(hypot)
#     hypot{T<:Array,S<:Array}(x1::Node{T},x2::Node{S})=hypot_r(x1,x2)
#     hypot{T<:Array,S<:Array}(x1::Node{T},x2::S)=hypot_r(x1,x2)
#     hypot{T<:Array,S<:Array}(x1::T,x2::Node{S})=hypot_r(x1,x2)

# I wrote the @primitive macro in util.jl to automate this process.
# One restriction is the inability to target parametric methods such
# as `f{T<:Number}(AbstractArray{T})`.  Julia does not support
# `f{T<:Number,A<:AbstractArray{T}}(Node{A})` yet.

# One could also choose to be lazy and just say:

#     hypot(x...) = hypot_r(x...)

# This would send any argument combination not covered by regular
# hypot methods to the recorder function, which presumably includes
# Noded calls.  This is dangerous for several reasons: (1) the Julia
# base may contain a typeless method (e.g. it does for `vcat`) we are
# overwriting. (2) this catches Noded calls to hypot methods we may
# not support yet.  So generally I would not recommend it.

# 6.2 Gradients

if !isdefined(:Grad)
    immutable Grad{N}; end          # Gradient wrt N'th argument
end

# In AutoGrad, gradients are represented by high-order gradient maker
# functions for each argument of each primitive method.  A gradient
# maker takes an argument specifier `Grad{N}`, the return value `y`,
# and the input arguments `x...`, and returns a gradient function (a
# closure) that turns `dJ/dy` into `dJ/dx_N`.  For the first example
# here is the gradient maker:

# `sin{T<:Number}(::Type{Grad{1}}, ::Node, x::Node{T})=(dy->dy*cos(x))`

# Note that the parameters and the return variable of the original
# function can be used in the gradient function.  For the second
# example a different gradient maker is generated for each argument:

# `hypot{T<:Array,S<:Array}(::Type{Grad{1}},y::Node,x1::Node{T},x2::Node{S})=(dy->dy.*x1./y)`
# `hypot{T<:Array,S<:Array}(::Type{Grad{2}},y::Node,x1::Node{T},x2::Node{S})=(dy->dy.*x2./y)`

# And of course we need four more definitions for the other
# boxed/unboxed argument combinations, which the @primitive macro
# generates automatically.

# Zero gradient functions such as `sign`, and non-numeric functions
# such as `size` should be defined using the @zerograd macro instead.
# Unlike primitives, zerograd functions neither record their action
# nor return a Node.  They just unbox their arguments and return a
# regular value.

# Finally some methods such as `sum(a::Array,i::Int)` are only
# differentiable wrt some of their arguments (here `a` but not `i`).
# These methods must record when their differentiable argument(s) are
# boxed and return boxed values.  The gradient makers for the other
# arguments can be left undefined (if they cannot be boxed as in the
# sum example), or defined as returning 0 instead of a gradient
# function (if they can be boxed, see ungetindex).


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

getval(x) = isa(x, Node) ? x.value : x  # we never create Node(Node).

# 8.2 merge_tapes: Used by forward_pass to support higher order
# gradients. Records its operation if both its arguments are Nodes.
# Its first argument is a newly created Node.  Its second argument is
# the original input argument, which is only a Node if this is a
# higher order gradient, in which case it is a Node that belongs to
# another tape and merge_tapes in effect creates a third Node on both
# tapes that point to their respective parent Nodes.  If the second
# argument is not a Node, simply returns its first arg without
# recording anything.

merge_tapes(a,b)=a
merge_tapes_r = recorder(merge_tapes)
merge_tapes{T}(a::Node{T},b::Node{T})=merge_tapes_r(a,b)
# The derivatives simply pass the gradients back.
merge_tapes{N,T}(::Type{Grad{N}},c::Node{T},a::Node{T},b::Node{T}) = identity


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


# 8.4 sum_outgrads: only used in backward_pass to produce the input to
# a gradient function df/dy.  Does it ever need to record its
# operation and give a Node output?  Gradient functions are closures
# that take df/dy, return df/dx and have the environment (x,y) which
# is the original input/output and almost certainly Nodes.  The input
# df/dy may or may not be a Node.  If we encounter a Node with an open
# tape, we need to record the sum operation.  This will happen in
# higher order derivatives.

# A: what if first outgrad is a Node and others aren't, or vice versa?  handled by @primitive.
# A: what should the output be if some of the inputs are Nodes?  Is sum_outgrads a primitive? No, its input is never a Node, maybe an array including Nodes.  Its helper is a primitive.
# A: for array and dict can we just modify the first element of outgrads?  This may be dangerous because the same outgrad may be passed back to more than one target e.g. by `+`.

# TODO: Instead of maintaining an array of outgrads then summing them, why not keep a sum to avoid allocation?
# (instead of pushing to parent.outgrads, we'd have to call sum directly, deal with Nodes etc.)
# However before we can do this we need to handle overwriting ops because sum_outgrads is a primitive.

function sum_outgrads(x)
    length(x) == 0 && return nothing
    length(x) == 1 && return x[1]
    a = remove_nothings(x)
    length(a) == 0 && return nothing
    length(a) == 1 && return a[1]
    sum_helper(a...)
end

sum_helper(a::Number, b::Number, c::Number...)=sum([a,b,c...])
sum_helper(a::Tuple, b::Tuple, c::Tuple...)=tuple([sum_outgrads(e) for e in zip(a,b,c...)]...)
sum_helper{T}(a::AbstractArray{T},b::AbstractArray{T},c::AbstractArray{T}...) =
    (isbits(T) ? broadcast(+,a,b,c...) : [sum_outgrads(e) for e in zip(a,b,c...)])
sum_helper(a::Associative, b::Associative, c::Associative...) =
    (z=similar(a); for d in (a,b,c...), (k,v) in d; z[k]=v+get(z,k,0); end; z)

sum_helper_r = recorder(sum_helper)
sum_helper(x...)=sum_helper_r(x...)
sum_helper{N}(::Type{Grad{N}}, y, x...)=identity

function remove_nothings(x)
    a = []
    for xi in x
        isa(xi,Void) || push!(a,xi)
    end
    return a
end

# TODO: check if we really need tofloat.
# converts nested values to float.
# tofloat uses float for Number and AbstractArray (for isleaftype)
tofloat(x::Number)=float(x)
tofloat{T<:Number}(x::AbstractArray{T})=float(x)
# tofloat extends to Tuple, Associative, and arbitrary Arrays
tofloat(x::Tuple)=(all(isfloat,x) ? x : ntuple(i->tofloat(x[i]), length(x)))
tofloat(x::Associative)=(all(isfloat,values(x)) ? x : (a=similar(x); for (k,v) in x; a[k]=tofloat(v); end; a))
tofloat(x::AbstractArray)=(all(isfloat,x) ? x : map(tofloat,x))
isfloat(x)=isa(x,AbstractFloat)
