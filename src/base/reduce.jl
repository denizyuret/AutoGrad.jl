reduce1sum = Dict{Symbol,Any}(
:sum => :(dy->dy.+zeros(x)),
)
defgrads(reduce1sum, AbstractArray; dymul=false)

reduce2sum = Dict{Symbol,Any}(
:sum => (:(dy->dy.+zeros(x1)),0)
)

@primitive sum{T<:Integer}(x1::BitArray,x2::Node{T}) # To avoid clash with bitarray.jl:1501.
defgrads(reduce2sum, AbstractArray, Integer; dymul=false)

testargs{T1<:AbstractArray,T2<:Number}(::Fn{:sum}, ::Type{T1}, ::Type{T2})=(randn(2,2),1)

Base.zeros(x::Node)=zeros(x.value)

# TODO: implement more general sum ops
