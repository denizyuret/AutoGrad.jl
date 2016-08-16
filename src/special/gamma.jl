gamma1arg = Dict{Symbol,Any}(
:digamma => :(trigamma(x)),  # gamma,operators = polygamma(0,x)
#:eta => :todo,      # gamma,operators
#:gamma => :todo, # gamma,operators
#:invdigamma => :todo, # gamma,operators
#:lfact => :todo,  # gamma,operators
:trigamma => :(polygamma(2,x)), # gamma,operators = polygamma(1,x)
#:zeta => :todo,  # gamma,operators
)

defgrads(gamma1arg, Number)
defgrads(gamma1arg, AbstractArray)

gamma2arg = Dict{Symbol,Any}(
:beta => :todo,                          # gamma,operators
:lbeta => :todo,                         # gamma,operators
:polygamma => (0,:(polygamma(x1+1,x2))), # first argument should be an integer; gamma,operators
:zeta => :todo,                 # domain >= 1?; gamma, operators
)

defgrads(gamma2arg, Number, Number)
defgrads(gamma2arg, AbstractArray, Number)
defgrads(gamma2arg, Number, AbstractArray)
defgrads(gamma2arg, AbstractArray, AbstractArray)

testargs{T1<:Number,T2}(::Fn{:polygamma}, ::Type{T1}, ::Type{T2})=(rand(0:5),testargs(Fn2(:polygamma),T2)...)
testargs{T1<:AbstractArray,T2}(::Fn{:polygamma}, ::Type{T1}, ::Type{T2})=(rand(0:5,2),testargs(Fn2(:polygamma),T2)...)
