# module AutoGrad

importall Base  # defining getindex, sin, etc.
# importall Knet  # quadloss etc.

isdefined(:DBG) || (DBG=Dict())
dbg(k,x)=get(DBG,k,false)&&println(k,": ",x)
dbg(k)=(DBG[k]=!get(DBG,k,false))

include("core.jl")
include("util.jl")
include("collections.jl")
include("math1arg.jl")
include("math2arg.jl")

# end # module
