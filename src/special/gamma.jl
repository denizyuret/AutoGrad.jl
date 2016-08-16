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
:polygamma => :todo,            # first argument should be an integer; gamma,operators
:zeta => :todo,                 # domain >= 1?; gamma, operators
)
