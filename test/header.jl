using Compat
using Compat.Test, Compat.LinearAlgebra
pushfirst!(LOAD_PATH, joinpath(dirname(@__FILE__),"../src"))
using AutoGrad
