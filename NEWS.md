AutoGrad v1.1.6 Release Notes
=============================

* Gradients returned can now be of type AutoGrad.Sparse to make large lookup parameters more efficient.
* Refactoring: UngetIndex -> Sparse, sum_outgrad -> addto!, Tape.list back to normal order.
* Timer messages include tape position so we can see which op takes how much time forw/back.


AutoGrad v1.1.5 Release Notes
=============================
ca0d03f 2019-09-20

* Fixed single-argument permutedims.
* SpecialFunctions lgamma->loggamma transition.
* Support for stdm and varm.
* Typo fix in params.
* gcnode now takes Tape as a second argument.


AutoGrad v1.1.4 Release Notes
=============================
0a4fcd2 2019-05-25

* High order gradient fixes.


AutoGrad v1.1.3 Release Notes
=============================
c80fa9c 2019-01-20

* Use `broadcasted` rather than `broadcast` as primitive, allows `p .-= g` where p::Param.
* Define efficient params(::Tape).
* doc updates.


AutoGrad v1.1.2 Release Notes
=============================
deeb980 2019-01-04

* Support for broadcasting user defined functions.
* gcheck and @gcheck for gradient checking with Params. (with @ekinakyurek)
* Added @primitive2 and @zerograd2 for broadcast-only primitives.
* Handle functions where result is not the last thing on tape.


AutoGrad v1.1.1 Release Notes
=============================
0ec058b 2018-09-30

* General performance improvements.
* Val()->Arg{}
* Result(x, f, args...; kwargs...)->Result(x, f, args, kwargs)
* Improved display for Tapes.
* Using TimerOutputs to profile core.jl. Use AUTOGRAD_TIMER environment variable.
* Added gcnode() to save memory, overriding in Knet to free GPU arrays.
* Added Node.children to be used by gcnode in Knet.


AutoGrad v1.1.0 Release Notes
=============================
5642caf 2018-09-05

* Solved tape confusion issue.
* Allow setindex! on Values if not differentiating.
* Added gradloss.
* Improved display for Array Params.
* Added the @diff macro, preferred to the differentiate function.
* Added the opt field to Param.
* Switch to using grad instead of gradient.

AutoGrad v1.0.1 Release Notes
=============================
c3a91a8 2018-08-25

* Using forw and back methods instead of the recorder function generator and the Grad type.
* New gradcheck method checks all arguments, handles tuples and dicts as well as arrays.
* Memory saving by erasing outgrads from tape after use (@ekinakyurek).
* Fixed integer powers and matrix powers.
* Fixed sum_outgrads_array bug causing incorrect grad when indexing into a matrix of vectors #73.
* Fixed sum_outgrads bug so arrays of different types can be added #71.
* Robust Float32 testing added #87 (@ekinakyurek).
* Cleaned up and documented the handling of broadcasting.
* Codecov support added.

AutoGrad v1.0.0 Release Notes
=============================
bb5e356 2018-08-19

* Julia 1.0 compatibility release (pre-0.7 Julia versions no longer supported) (@rened, @CarloLucibello).
* Higher order gradient bug fixed (@MikeInnes).
* Improved unit tests (@ekinakyurek).


AutoGrad v0.1.0 Release Notes
=============================
b6b5863 2018-05-30

* Pre-0.6 Julia versions no longer supported.
* LinAlg support (@CarloLucibello): det, logdet, inv, dot, diag, diagm, trace, logabsdet, kron, triu, tril, chol, lq, qr, svd
* Added grads for std, var, Int^Int (@CarloLucibello)
* Added f(Rec(::Type{T}) = f(T) for eltype, ndims, one, zero (@CarloLucibello)
* Fixed reduction of scalars, e.g. `grad(sum)(3.1)` (@CarloLucibello)
* Fixed `similar(::Rec{KnetArray}, ::Dims)`


AutoGrad v0.0.10 Release Notes
==============================
823ea16 2017-11-28

* Fixed gradient output type for erf family (@CarloLucibello).
* Added mean(x,dims) (@CarloLucibello).
* Added support for Dict iteration.
* Added profiling support inspired by TimerOutputs.jl.


AutoGrad v0.0.9 Release Notes
=============================
0c27dd8 2017-10-18

* Added SpecialFunctions for Julia 0.6+.
* Fixed test problem with broadcast#log.
* Implemented cat1d, an efficient cat function for many arguments.


AutoGrad v0.0.8 Release Notes
=============================
2f9f7f3 on 2017-09-08

* Julia v0.6.0 compatibility fixes by @ylxdzsw #24.  Now compatible with Julia v4, v5, and v6.
* Switched to new Base.Test functionality added in Julia v0.5.
* Fix size(rec,d1,d2,ds...) from @CarloLucibello #19.
* Fix depwarn from @mdpradeep #23.
* Generalize cat implementation to work with any number of arguments.
* Fix sign bug in derivatives for minabs/maxabs.


AutoGrad v0.0.7 Release Notes
=============================
71a99da on 2017-05-17

* Fixed getindex bug effecting repeated indices.

* Fixed minor bug in `@zerograd` which did not handle some type
  declarations.

* `dumptape()` and `AutoGrad.debugtape(::Bool)` utilities for
  debugging.

AutoGrad v0.0.6 Release Notes
=============================
(e0530fe  on Feb 23, 2017)

* Fixed typo effecting broadcast comparison operators like .<

* Added scalar support to Ac_mul_B etc. which were causing trouble
  with expressions like Array'*Scalar.

AutoGrad v0.0.5 Release Notes
=============================
(558a217 on Feb 13, 2017)

* gradloss is added as an alternative to grad.  gradloss(f) generates
  a function that returns a (gradient, result) tuple.

* gradcheck is added as an alternative gradient checking mechanism. It
  performs sampling to handle large arrays, and automatically
  constructs a scalar function to handle array functions.

* Added gradients for vecnorm, (.!=), (\), linearindexing, vec,
  permutedims, ipermutedims.

* `sum_outgrads{T}(a::AbstractArray{T},b::AbstractArray{T})` was not
  type preserving, now fixed.


AutoGrad v0.0.4 Release Notes
=============================
(f7aaac8 on Oct 23, 2016)

* The word value was being used for too many things.  The type Value
  has been renamed Rec.  The field Node.value got renamed to Node.rec.

* Added reshape to gradients of matmul, unbroadcast etc.  Julia array
  math handles missing rightmost dimensions as 1 so for example
  rand(1,2)*rand(2) works.  Gradients should be careful to preserve
  the input shape.

* vcat and hcat are now defined in terms of cat and they should handle
  arrays of any dimensionality and gradients should preserve the
  original input shape.

* getval and fixdomain not exported any more.  fixdomain is
  deprecated.  getval should not be necessary because Recs should
  never be exposed outside AutoGrad.



