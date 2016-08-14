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

bessel2arg = Dict{Symbol,Any}(
#:airy => :todo,                  # first arg should be an integer; bessel,operators
#:airyx => :todo,                 # first arg should be an integer; bessel,operators
:besselh => :todo,                       # bessel,operators
:besseli => :todo,                       # bessel,operators
:besselix => :todo,                      # bessel,operators
:besselj => :todo,                       # bessel,operators
:besseljx => :todo,                      # bessel,operators
:besselk => :todo,                       # bessel,operators
:besselkx => :todo,                      # bessel,operators
:bessely => :todo,                       # bessel,operators
:besselyx => :todo,                      # bessel,operators
:hankelh1 => :todo,                      # bessel,operators
:hankelh1x => :todo,                     # bessel,operators
:hankelh2 => :todo,                      # bessel,operators
:hankelh2x => :todo,                     # bessel,operators
)
