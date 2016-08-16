bessel1arg = Dict{Symbol,Any}(
:airy => :(airyprime(x)),   # airy(z)=airy(0,z); airy(k,x): The kth derivative of the Airy function Ai(x); bessel,operators
:airyai => :(airyprime(x)), # airyai(z)=airy(0,z); bessel,operators
#FAIL :airyaiprime => :(airybi(x)), # airyaiprime(z)=airy(1,z); bessel,operators
:airybi => :(airybiprime(x)), # airybi(z) = airy(2,z); bessel,operators
#FAIL :airybiprime => :(airy(4,x)), # airybiprime(z) = airy(3,z); bessel,operators
#FAIL :airyprime => :(airybi(x)),  # airyprime(z)=airy(1,z); bessel,operators
#:airyx => :todo, # airyx(z)=airyx(0,z); bessel,operators
#:besselj0 => :todo,		# bessel,operators
#:besselj1 => :todo,		# bessel,operators
#:bessely0 => :todo,		# bessel,operators
#:bessely1 => :todo,		# bessel,operators
)

defgrads(bessel1arg, Number)
defgrads(bessel1arg, AbstractArray)

bessel2int = Dict{Symbol,Any}(
:airy => (0,:(airy(x1+1,x2))), # TODO: cannot handle single nondifferentiable arg but we need 2-arg airy to accept Nodes, so this is a temporary workaround for now. # (0,:(airy(x1+1,x2))), # first arg should be an integer; bessel,operators
# :airyx => :todo,                 # first arg should be an integer; bessel,operators
)

defgrads(bessel2int, Int, Number)
defgrads(bessel2int, AbstractArray{Int}, Number)
defgrads(bessel2int, Int, AbstractArray)
defgrads(bessel2int, AbstractArray{Int}, AbstractArray)

# k must be between 0 and 3 so we test with 0:2 since grad requires k+1
testargs(::Fn{:airy},k,x) =
    ((k<:AbstractArray ? rand(0:2,2) : rand(0:2)),
     (x<:AbstractArray ? randn(2) : randn()))

bessel2arg = Dict{Symbol,Any}(
# :besselh => :todo,                       # bessel,operators
# :besseli => :todo,                       # bessel,operators
# :besselix => :todo,                      # bessel,operators
# :besselj => :todo,                       # bessel,operators
# :besseljx => :todo,                      # bessel,operators
# :besselk => :todo,                       # bessel,operators
# :besselkx => :todo,                      # bessel,operators
# :bessely => :todo,                       # bessel,operators
# :besselyx => :todo,                      # bessel,operators
# :hankelh1 => :todo,                      # bessel,operators
# :hankelh1x => :todo,                     # bessel,operators
# :hankelh2 => :todo,                      # bessel,operators
# :hankelh2x => :todo,                     # bessel,operators
)

