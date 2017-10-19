AutoGrad v0.0.10 Release Notes
==============================

* Fixed gradient output type for erf family (@CarloLucibello).


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



