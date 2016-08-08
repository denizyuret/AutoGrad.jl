# Contents:

# Here are the rough steps performed by g(x) where g=grad(f):
# 1. g is called with the same inputs as f.
# 2. g calls forward_pass which boxes x in a Node type and calls f(Node(x)).
# 3. If a primitive operator inside f gets a Node input, it records its action and returns a Node output.
# 4. g calls backward_pass which returns the gradient df/dx.

# And some background info:
# 5. The types used in recording: Node, ReverseNode, CalculationTape.
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
    dbg((:grad,name(gradfun,(symbol("D$argnum"),name(fun)))))
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
    dbg((symbol("forw$argnum"), name(fun), args..., kwargs...))
    tape = CalculationTape()
    arg_wrt = args[argnum]
    start_node = new_node(safe_type(getval(arg_wrt)), Any[tape])
    args = Any[args...] # to make args writeable
    args[argnum] = merge_tapes(start_node, arg_wrt)
    end_node = fun(args...; kwargs...)
    return start_node, end_node, tape
end

# forward_pass: ((N(X)->N(Y)/Y),N(X)/X,K,I)->(N(X),N(Y)/Y,T)
# deps: CalculationTape, new_node, safe_type, getval, merge_tapes
new_node(value, tapes)=Node(value, tapes)
safe_type(value) = isa(value, Integer) ? float(value) : value
getval(x) = isa(x, Node) ? x.value : x  # we never create Node(Node).


# 3. If a primitive operator inside f gets a Node input, it records its action and returns a Node output.

# 3.1 We would like primitives to record what they are doing and
# return a Node if any of their arguments is a Node (exceptions:
# gradient functions, utilities?).  One way to do this is to define a
# catch-all method p(x...; o...) for primitive p which gets called
# when no existing methods match.  This breaks down if there is an
# existing method that handles ::Any arguments, so one must be
# careful.  In that case methods with more specific Node type
# signatures must be defined instead of using the @primitive macro.

macro primitive(f)
    esc(:(local r = recorder($f); $f(x...;o...)=r(x...;o...)))
end

# 3.2 This catch-all method is produced by r=recorder(f).

"""
recorder(fun) returns rfun, a recording version of fun.  rfun is defined with
a generic signature r(args...; kwargs...) and is intended to catch all
invocations that have at least one Node argument.
"""
function recorder(f)
    #dbg((:recorder,f))
    function r(args...; kwargs...)
        dbg((:call, name(f), args..., kwargs...))
        argvals = Any[args...]
        ops = []
        tapes = Set()
        found_node = false

# 3.3 r goes through the arguments and unboxes any that are Nodes.
# For each unboxed Node, its tape, argnum, and ReverseNode is stored.


        for (i, arg) in enumerate(args)
            if isa(arg, Node)
                found_node = true
                argvals[i] = arg.value
                # if i in p.zero_grads; continue; end               # Q: who sets zero_grads, why?  TODO: reimplement
                for (tape, parent_rnode) in arg.tapes               # Node.tapes is a Dict{Tape,ReverseNode}
                    if !iscomplete(tape)                            # why do we need iscomplete? to prevent recording during the backward_pass unless we are doing higher order derivatives.
                        push!(ops, (tape, i, parent_rnode))         # ops should be called args or inputs!
                        push!(tapes, tape)                          
                    end
                end
            end
        end

# 3.4 If no Nodes found we throw an error (the catch-all method was
# supposed to catch calls with Node arguments).

        found_node || throw(MethodError(f, argvals))            # Otherwise undefined methods lead to infinite loop

# 3.5 The primitive is called with unboxed arguments.
        
        result = f(argvals...; kwargs...)

# 3.6 ops can be empty if no Nodes, zero_grads, or iscomplete(tape).
# No Nodes case is impossible, we throw an error.
# Q: why do we need zero_grads?
# A: why do we need iscomplete? to prevent recording during backward_pass unless higher order derivatives.

        if !isempty(ops) 

# 3.7 We box the result in a Node attached to all the tapes we
# encountered in 3.3.  This boxed result gets returned.

            result = new_node(result, tapes)

# 3.8 For each of our Node inputs, we create a gradient function for
# the specific argnum of the primitive.  This gradfun takes dy and
# returns dx and has access to the original x,y through a closure.
# The gradfun and the ReverseNode of the associated input is stored in
# parent_grad_ops.

            for (tape, argnum, parent) in ops                       
                gradfun = nothing
                try 
                    gradfun = f(Val{argnum}, result, args...; kwargs...) # Creates a node specific gradfun (dy->dx) with x,y in a closure
                catch e
                    isa(e, MethodError) ? (warn("$e");continue) : throw(e)
                end
                name(gradfun,(symbol("D$argnum"),f,:out,result,:args,args...,kwargs...)) # Record for debugging
                rnode = result.tapes[tape]
                push!(rnode.parent_grad_ops, (gradfun, parent))
                dbg((:deps,name(tape),rnode))
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
    dbg((:back,name(start_node),name(end_node),name(tape)))

# 4.2 If end_node is not a Node on the given tape, we return zero df/fx.
# end_node may not be a Node if the output of f does not depend on x.
# Q: Could the end_node be a Node but in a different tape?    
# Q: Could df/dx be a Node?
# Q: Should zeros_like return a Node if the input is a Node?    

    if !isa(end_node, Node) || !haskey(end_node.tapes, tape)    # This may happen e.g. if the function returns a constant
        warn("Output seems independent of input. Returning zero gradient.")
        return zeros_like(start_node)
    end

# 4.3 backward_pass resets all node gradients except for the scalar
# output Node whose gradient is set to 1.0.
# Q: Why do we need complete!(tape)?

    for node in tape                                            # tape is created by forw_pass
        node.outgrads = []
    end
    end_node.tapes[tape].outgrads = [1.0]                       # end_node.tapes[tape] is the ReverseNode corresponding to end_node::Node

    complete!(tape)

# 4.4 the tape is read in reverse and for each node with a non-empty
# outgrad its ingrads are computed using the closures recorded by the
# primitive recorders.

    cur_outgrad = nothing
    for node in tape[end-1:-1:1]                                # note the end-1 because we pushed nothing to complete
        if !isempty(node.outgrads)
            cur_outgrad = sum_outgrads(node.outgrads...)
            dbg((:sum,name(node),cur_outgrad,node.outgrads...))
            for (gradfun, parent) in node.parent_grad_ops
                dbg((:back1,cur_outgrad,name(gradfun)))
                og = gradfun(cur_outgrad)
                push!(parent.outgrads, og)
                dbg((:back2,og,name(gradfun)))
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
# empty, thus only the necessary gradients are computed.

    return cur_outgrad
end


# 5. The types used in recording: Node, ReverseNode, CalculationTape.

"""
Node(value, tapes) creates a new Node:

1. in forward_pass for the argument we are taking gradient w.r.t.
2. for the output of a primitive operation with Node input.

When a Node is created, it pushes ReverseNodes with the same value on
each of the tapes given in the second argument, and records pointers
to each tape and its ReverseNode in its `tapes` dictionary.  These
ReverseNodes have empty parent_grad_ops and outgrads which are written
by call and back respectively.  Ordinarily there is only one tape
(unless we do higher order derivatives).
"""
type Node; value; tapes;
    function Node(value, tapes)                                 # arg tapes is an Array
        self = new(value, ObjectIdDict())                       # field tapes is an ObjectIdDict{CalculationTape,ReverseNode}
        for tape in tapes
            new_rnode = ReverseNode(typeof(self), value)        # Q: do we need typeof here any more if we have a single Node type?  also why define self.Rnode in python?
            push!(tape, new_rnode)                              # This is the only place new elements are added to a tape.
            self.tapes[tape] = new_rnode
        end
        dbg((:node,self))
        return self
    end
end

"""
ReverseNode is a plain type with four slots:

* `parent_grad_ops`: `call` fills this array with (gradfun,parent_rnode) pairs for each Node argument.
* `outgrads`: used by backward_pass, array of gradients for this node (has multiple elements if fanout > 1).
* `node_type`: type of corresponding Node
* `node_value`: same value as corresponding Node
"""    
type ReverseNode; parent_grad_ops; outgrads; node_type; node_value; end
ReverseNode(node_type, node_value) = ReverseNode([], [], node_type, node_value)

"CalculationTape is an array of ReverseNodes with a `complete` flag."
typealias CalculationTape Array{ReverseNode,1}
iscomplete(a)=(!isempty(a) && a[end].node_value==nothing)
complete!(a)=push!(a,ReverseNode(nothing,nothing))


# 6. How new primitives and their gradients are defined.

# See #3 for defining primitives.  For gradients we define a gradmaker
# for each primitive method p and argnum: a function that takes (x,y)
# (the argnum'th input and output of p) and returns a gradient
# function (df/dy->df/dx).  Julia has multiple-dispatch, which means
# each argument type combination for a function might end up calling a
# different method, each potentially requiring different gradients.
# So we store gradmakers in methods that take (Val{N}, y, x...).  This
# way we can use method dispatch to find the appropriate gradient.
# Example: sin(::D1,y,x)=(dfdy->dfdy*cos(x))

typealias D1 Type{Val{1}}
typealias D2 Type{Val{2}}
typealias D{N} Type{Val{N}}

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
# Nodes and record their gradients.

# 8.1 new_node, getval, safe_type: these box, unbox, and float values.

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

# Problem: If we use regular @primitive, merge_tapes(x,y) overrides merge_tapes(x...)
# So we define recorder for merge_tapes manually
merge_tapes_r = recorder(merge_tapes)
merge_tapes(x::Node,y::Node) = merge_tapes_r(x,y)

merge_tapes(::D1,c,a,b) = (x->x)   
merge_tapes(::D2,c,a,b) = (x->x)

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

# 8.4 zeros_like: used by backward_pass and getindex.  Should it
# return a Node when its input is a Node?

# In backward_pass only used to give the return value when end_node is
# not a Node on the given tape.  This means the output of f is
# independent of input.  In a simple gradient, backward_pass always
# returns a value.  In a high-order gradient, backward_pass(f) returns
# a Node, and backward_pass(g) returns a value.  However even in
# backward_pass(f) we do not need to return a Node if the value we are
# returning is a constant that does not depend on the input!  So
# zeros_like can always return a regular value.

# In getindex(::D1) zeros_like is never called with a Node input.  So
# it is safe to unbox the input of zeros_like and not box its result.

"""
zeros_like(x) -> value or object similar to x filled with zeros.
Can handle bits types, Array, Tuple, Associative, and Node.
Implementation similar to deepcopy.
TODO: avoid allocating large arrays using `nothing` like Knet.
"""
zeros_like(x) = zeros_internal(x, ObjectIdDict())
zeros_check(x, d::ObjectIdDict)=(haskey(d,x) ? d[x] : d[x]=zeros_internal(x,d))
# Q: should the output be wrapped in Node if the input is a Node?
zeros_internal(x::Node,d::ObjectIdDict)=zeros_check(x.value,d)
zeros_internal(x::Tuple,d::ObjectIdDict)=ntuple(i->zeros_check(x[i],d), length(x))
zeros_internal(x::Associative,d::ObjectIdDict)=[ k => zeros_check(v,d) for (k,v) in x ]
zeros_internal{T}(x::AbstractArray{T},d::ObjectIdDict)=(isbits(T) ? zeros(x) : T[zeros_check(e) for e in x])
zeros_internal{T}(x::T,d::ObjectIdDict)=(isbits(T) ? zero(x) : error("zeros_like cannot handle $T"))


# 8.5 sum_outgrads: only used in backward_pass to produce the input to
# a gradient function df/dy.  Does it ever need to record its
# operation and give a Node output?  Gradient functions are closures
# that take df/dy, return df/dx and have the environment (x,y) which
# is the original input/output and almost certainly Nodes.  The input
# df/dy may or may not be a Node.  If we encounter a Node with an open
# tape, we need to record the sum operation.  This will happen in
# higher order derivatives.

# TODO: Instead of maintaining an array of outgrads then summing them, why not keep a sum to avoid allocation?
# (instead of pushing to parent.outgrads, we'd have to call sum directly)
# Q: for array and dict we are modifying the first element of outgrads, is that ok?
# Q: what if first outgrad is a Node and others aren't, or vice versa?
# Q: what should the output be if some of the inputs are Nodes?  Is sum_outgrads a primitive?

sum_outgrads(x)=x
sum_outgrads(a::Number, b::Number, c::Number...)=sum([a,b,c...])
sum_outgrads(a::Tuple, b::Tuple, c::Tuple...)=tuple([sum_outgrads(e...) for e in zip(a,b,c...)]...)
sum_outgrads{T}(a::AbstractArray{T},b::AbstractArray{T},c::AbstractArray{T}...) =
    (isbits(T) ? broadcast!(+,a,a,b,c...) : [sum_outgrads(e...) for e in zip(a,b,c...)])
sum_outgrads(a::Associative, b::Associative, c::Associative...) =
    (for d in (b,c...), (k,v) in d; a[k]=v+get(a,k,0); end; a)
sum_outgrads{N}(::D{N}, y, x...)=(dy->dy)
@primitive sum_outgrads

# Container types are handled by overloading getindex:
# Top level Julia container types that support getindex:
# Associative, AbstractArray, Tuple
# getindex(obj, key...) => value
# setindex!(obj, val, key...) => obj

importall Base

@primitive getindex
getindex(::D1,y,x,i...) = dy->ungetindex(x,dy,i...) # y=x[i]
ungetindex(x::AbstractArray, dy, i...) = (z=zeros_like(x);setindex!(z,dy,i...);z)
ungetindex(x::Associative, dy, i...)   = (z=zeros_like(x);setindex!(z,dy,i...);z)
ungetindex(x::Tuple, dy, i)            = ntuple(j->(j==i ? dy : zeros_like(x[j])), length(x))

# Now we need zero grads!?  Try detecting and ignoring undefined gradmakers.
ungetindex(::D2, out, x, dy, i...) = g->getindex(g,i...)
@primitive ungetindex

# Pretty print for debugging:
dbg(x)=println(x)
_name=ObjectIdDict()
name(f,n)=(_name[f]=n)
name(f)=get(_name,f,f)
name(x::ReverseNode)=symbol("R$(href(x))")
name(x::Node)=symbol("N$(href(x))")
name(x::Array)=symbol("A$(href(Ref(x)))")
name(x::Tuple)=map(name,x)
href(x)=Int(hash(x)%100)

Base.show(io::IO, n::Node) = print(io,"$(name(n))$((n.value,[(name(t),name(r)) for (t,r) in n.tapes]...))")
Base.show(io::IO, n::ReverseNode) = print(io,"$(name(n))$((n.node_value,n.outgrads,[(name(y),name(x)) for (x,y) in n.parent_grad_ops]...))")

# Examples:
@primitive(sin)
@primitive(cos)
@primitive(+)
@primitive(*)
@primitive(-)

# Q: alt notation: sin(x...,y,:D1) or sin(x...,y,dy,:D1) for non-closure interface
# however this does not allow x... in definition which may be useful.
sin(::D1,y,x)=(dy->dy*cos(x))
cos(::D1,y,x)=(dy->dy*(-sin(x)))
(+)(::D1,y,x1,x2)=(dy->dy)
(+)(::D2,y,x1,x2)=(dy->dy)
(*)(::D1,y,x1,x2)=(dy->dy*x2)
(*)(::D2,y,x1,x2)=(dy->dy*x1)
(-)(::D1,y,x)=(dy->-dy)

# Examples:

function test1(x=(1.,2.))
    foo(x)=sin(x[1])+cos(x[2])
    goo = grad(foo)
    @show goo(x)
    hoo = grad(goo)
    @show hoo(x)
end

function test2()
    gsin = grad(sin)
    hsin = grad(gsin)
    #@show sin(1.0)
    #@show gsin(1.0)
    @show hsin(1.0)
end

function test3()
    foo2(x,y)=sin(x)+cos(y)
    goo2 = grad(foo2)
    goo22 = grad(foo2, 2)
    @show goo2(1,2)
    @show goo22(1,2)
end

# Q: Can we get away with a single Node type?
# Q: Why do we need multiple tapes?  Hypothesis: for higher level derivatives.
# Q: If we build a tree, how do we prune non-parameter inputs?  (they won't have node types)
# Q: How do we get derivatives for multiple parameters (possibly wrapped up in one list)?

# OK, at this point we get:
# MethodError: `sin` has no method matching sin(::Node)
# It is time to implement primitive.

# TODO: what is partial, define it?
# TODO: what is primitive with aux, do we need it?
# TODO: NotImplemented, NoDerivativeNode?
# DONE: zeros_like

# function gradmaker(p::Primitive, argnum, ans, args, kwargs)
#     try 
#         p.grads[argnum](ans, args...; kwargs...)
#     catch e
#         if isa(e, KeyError)
#             name = p.fun.env.name
#             if isempty(p.grads)
#                 error("Gradient of $name not yet implemented.")
#             else
#                 error("Gradient of $name w.r.t. arg number $argnum not yet implemented.")
#             end
#         else
#             throw(e)
#         end
#     end
# end

# defgrad(p::Primitive, gradmaker, argnum=1) = (p.grads[argnum] = gradmaker)
# defgrads(p::Primitive, gradmaker, argnums) = (for argnum in argnums; defgrad(p, partial(gradmaker, argnum), argnum); end)
# defgrad_is_zero(p::Primitive, argnums=(1,))= (for argnum in argnums; push!(p.zero_grads, argnum); end)



# Type signatures for gradients:
# f: X->Y
# Primitive(f)=P: Node(X)->Node(Y)?
# gradmaker: (Y,X)->(dY->dX)
# grad: P -> (X->dX)
# defgrad: gradmaker -> Void (puts gradmaker in grads)

# Primitive definition:
# Q: should we ask for gradients right away?
# Q: can we do gradients without higher order functions?
# Q: we need a Node method, what else do we absolutely need?

# _psin = Primitive(sin)
# _pcos = Primitive(cos)
# sin(x::Node)=_psin(x)
# cos(x::Node)=_pcos(x)
# # Instead of recording inputs and outputs explicitly, we hide it in closures for every call?
# # We store pointers to all inputs and outputs!
# defgrad(_psin, (y,x)->(dy->dy * _pcos(x)))                         # dJ/dx = dJ/dy * dy/dx
# defgrad(_pcos, (y,x)->(dy->-dy * _psin(x)))

# (start_node, end_node, tape) = forward_pass(sin, [1.0,], [], 1)
# #show start_node # value:1.0, tapes:(tape=>tape[1])
# #show end_node   # value:0.84, tapes:(tape=>tape[2])
# #display(tape)
# # type ReverseNode; parent_grad_ops; outgrads; node_type; node_value; end
# # tape[1]: ReverseNode(Any[],Any[],Node,1.0)                                                                       
# # tape[2]: ReverseNode(Any[((anonymous function),ReverseNode(Any[],Any[],Node,1.0))],Any[],Node,0.8414709848078965)

# _gsin = grad(sin)
# _gcos = grad(cos)
# #show sin(1.0)
# #show _gsin(1.0)
# #show cos(1.0)
# #show _gcos(1.0)

# # Implement + for multi-input example:
# _padd = Primitive(+)
# defgrad(_padd, (c,a,b)->(g->g), 1)
# defgrad(_padd, (c,a,b)->(g->g), 2)
# (+)(a::Node,b::Node)=_padd(a,b)

# # foo1(x)=sin(sin(x)+cos(x))
# # (start_node, end_node, tape) = forward_pass(foo1, [1.0,], [], 1)
# # goo1 = grad(foo1)
# # @show foo1(1.0)
# # @show goo1(1.0)

# # foo3(x)=(a=sin(x);println(typeof(a));b=cos(x);return a)
# # goo3 = grad(foo3)
# # @show goo3(1.0)


# Q: who does array allocation for array gradients?

# Do we need this?  work-in-progress...
# 
# """
# ans, x correspond to the original output and input.
# gradfun is a function that takes dJ/dy and returns dJ/dx for non-arrays.
# """
# function unbroadcast(ans, x, gradfun, broadcast_idx=1)
#     # x is the argument that we're differentiating with respect to.
#     if isa(x, Array)
#         shape = size(x)
#         function new_fun(g)<
#             result = gradfun(g)
#             while anp.ndim(result) > len(shape)
#                 result = anp.sum(result, axis=broadcast_idx)
#             end
#             for axis, size in enumerate(shape)
#                 if size == 1
#                     result = anp.sum(result, axis=axis, keepdims=True)
#                 end
#             end
#             assert anp.shape(result) == shape
#             return result
#         end
#     elseif isarray(ans)
#         new_fun(g) = sum(gradfun(g))
#     else
#         return gradfun
#     end
#     # new_fun.__name__ = "unbroadcast_{0}".format(gradfun.__name__)
#     return new_fun
# end

:ok

# Q: how does this handle a scalar input?

# Q: how does this handle multiple scalar inputs?

# Q: how does this handle arrays?

# Q: how do we deal with multi-input functions?  From numpy_grads.py:
# anp.multiply.defgrad(lambda ans, x, y : unbroadcast(ans, x, lambda g : y * g))
# anp.multiply.defgrad(lambda ans, x, y : unbroadcast(ans, y, lambda g : x * g), argnum=1)
# However see make_grad_matmul in the same file.

# Q: instead of overwriting call and defining Primitive, can we just define methods for Node arguments?
# In that case we'd have to keep grads in a global hash.
# And we'd have to deal with multiple dispatch, i.e. multiple derivatives for multiple methods.
# Julia primitives like methods, which and invoke find the method of a function appropriate for given argtypes.
# methods(f) simply returns f.env which is a method table.

# Function: [:fptr,:env::MethodTable,:code]
# MethodTable: [:name,:defs::Method,:cache,:cache_arg1,:cache_targ,:max_args,:kwsorter,:module]
# Method: [:sig,:va,:isstaged,:tvars,:func,:invokes,:next]

# the hash could define a gradient for each method or each function (use Function and ObjectIdDict)
# there is also the problem of interface, autograd solves this with closures, we'd need to find a way to store input/output
# in autograd, there is no gradient but a gradient maker which is invoked with x,y and creates a function on demand


# """
# Wraps a function so that its gradient can be specified and its invocation
# can be recorded. For examples, see the docs.
# """
# type Primitive; fun; grads; zero_grads; end

# # Turn methods into functions that take Primitive as first arg?
# Primitive(fun) = Primitive(fun, Dict{Int,Function}(), Set{Int}())

# # This is where the main action is:
# # forward_pass wraps the relevant argument in a Node and calls the top level function.
# # Inside the function whenever a Primitive is called with a Node arg, we end up here.
# # call(p::Primitive) replaces arg Nodes with their values and calls p.fun.
# # for each Node arg, if it is not in zero_grads, and its tape is not complete:
# # we store (tape, i, parent_rnode) in array ops and keep track of unique tapes.
# # After the call, we wrap result in a node with all unique tapes.
# # for each (tape, i, parent_rnode) in ops:
# # we call gradmaker on p with argnum=i
# # we push (grad_p, parent_rnode) on parent_grad_ops of ReverseNode of result 

# function call(p::Primitive, args...; kwargs...)
#     argvals = Any[args...]
#     ops = []
#     tapes = Set()
#     for (i, arg) in enumerate(args)
#         if isa(arg, Node)
#             argvals[i] = arg.value
#             if i in p.zero_grads; continue; end                 # Q: who sets zero_grads, why?
#             for (tape, parent_rnode) in arg.tapes               # Node.tapes is a Dict{Tape,Node}
#                 if !iscomplete(tape)                            # Q: why do we need iscomplete? high-order derivatives?
#                     push!(ops, (tape, i, parent_rnode))         # ops should be called args or inputs!
#                     push!(tapes, tape)                          
#                 end
#             end
#         end
#     end
#     result = p.fun(argvals...; kwargs...)
#     # if isa(result, NotImplemented); return result; end        # Q: what's this for?  NotImplemented is a Python primitive!
#     if !isempty(ops)
#         result = new_node(result, tapes)
#         for (tape, argnum, parent) in ops                       
#             gradfun = gradmaker(p, argnum, result, args, kwargs) # Creates a node specific gradfun (dy->dx) with x,y in a closure by calling p.grads[argnum](y,x)
#             rnode = result.tapes[tape]
#             push!(rnode.parent_grad_ops, (gradfun, parent))
#         end
#     end
#     return result
# end

### Need to find a Julia way to register primitives and their gradients.
# - call Primitive if any of the args is Node
# -- if we define f(a...) and call back f if no Node do we get infinite loop?
# -- what if f(a...) already defined?
# - in Python they just replace the whole function with Primitive?
# - in Julia we only define a method.
# - how do we handle multiple gradient methods for a function with multiple methods?
# - we need a way to wrap built-in functions as well as user defined ones as Primitives
# - gradient can be another method of the same function that starts with a Gradient dummy type
# - need to decide on the call signature
# - in that case the Primitive also can be a method with a Forward dummy type -- no: internal calls are normal.
# - we have a function and its existing methods
# -- we need to add methods to handle Node arguments
# -- we need to define gradients of different methods
# -- do we add a forward method for each argtype?
# ++ or do we have a single fallback method to handle nodes?
# -- so it has to be f(a...), a single method that strips Nodes and calls the regular method.
# -- if no Nodes, does not call itself, just gives an error -> solves infinite loop.
# -- what if f(a...) already defined?
# - write now defgrad defines grad for one argnum of one Primitive.
# -- python functions are actually methods.
# -- but we may have multiple methods with different argtypes that have different gradients!
# -- so currently Primitive corresponds to a Method, not a Function.
# -- going forward this is no problem, we just strip Node and call regular function, record operation.
# -- we need gradients defined potentially for each argnum of each method
# -- the output could be an array.  output type does not determine method.  only input types do.
# -- so gradient or gradient maker method needs all inputs with their types!
# -- right now p.grads[argnum] stores one gradient maker per argument position.  It should at least be grads[argtypes,argnum].

# DONE:
# merge_tapes issue, how to define primitives
# Working with one Node type simplifies the code.
# Could be much simplified if we don't support higher derivatives:
# - no need for multiple tapes, merge_tapes.
# - no need for tape complete?
# Need to find a Julia way to register primitives and their gradients.
# Need to test second order derivatives.

# @primitive getindex

# Problem:
# julia> @which getindex(Val{1}, Node(1,[]), Node(1,[]), 1)
# getindex(T::Type{T}, vals...) at array.jl:165
# Again, not explicitly declaring Node causes issues.  Define recorder version manually.
# getindex_r = recorder(getindex)
# getindex(x::Node, i...)=getindex_r(x,i...)
# getindex(x...;o...)=error((:getindex,x...,o...))

# We also have composite types that can be used as structs:
# getfield(obj,key) => val
# setfield!(obj,key,val) => val
# @primitive getfield
# getfield(::D1,val,obj,key)=(g->(z=zeros_like(obj);setfield!(z,key,g);z))
# ERROR: Core.getfield cannot be extended
# ok, we give up for now (workaround would be to define own get function)
# function zeros_internal{T}(x::AbstractArray{T},d::ObjectIdDict)
#     haskey(d,x) && return d[x]
#     if isbits(T)
#         return (d[x]=zeros(x))
#     end
#     dest = similar(x)
#     for i=1:length(x)
#         if isdefined(x,i)
#             arrayset(dest, zeros_internal(x[i],d), i)
#         end
#     end
#     return (d[x]=dest)
# end

# function zeros_internal{T}(x::T,d::ObjectIdDict)
#     haskey(d,x) && return d[x]
#     nf = nfields(T)
#     (isbits(T) || nf == 0) && return zero(x)
#     y = ccall(:jl_new_struct_uninit, Any, (Any,), T)
#     for i in 1:nf
#         if isdefined(x,i)
#             ccall(:jl_set_nth_field, Void, (Any, Csize_t, Any), y, i-1,
#                   zeros_internal(getfield(x,i), d))
#         end
#     end
#     return (d[x]=y::T)
# end

# Q: return grad wrt all args instead of a single one?

# TODO:
# are closures efficient?
# Need to test arrays, tuples etc.  Handle getindex.
# Implement mnist examples.
# Implement general operators.
# Try gpu operations.
# Need to solve the allocation / overwriting problem.



    # #DBG
    # global _s,_e,_t
    # _s,_e,_t = start_node, end_node, tape
    # isa(end_node, Node) || warn("end_node is not a Node")
    # haskey(end_node.tapes, tape) || warn("cannot find tape")
    # #DBGEND
