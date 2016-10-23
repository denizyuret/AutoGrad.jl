AutoGrad v0.0.4 Release Notes
=============================

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



