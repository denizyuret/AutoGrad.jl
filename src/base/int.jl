int1arg = Dict{Symbol,Any}(
#:(~) => :todo, # bitwise not, domain=Integer; bool,int,gmp,bitarray,arraymath
)
int2arg = Dict{Symbol,Any}(
#:$ => :todo,                     # domain: Integers, bitwise xor; int,bool,bitarray,gmp,operators,promotion,arraymath,sparsematrix
#:& => :todo,                     # domain: Integers, bitwise and
#:| => :todo,                     # domain: Integers, bitwise or
)
