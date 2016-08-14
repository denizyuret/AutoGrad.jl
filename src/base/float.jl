float1arg = Dict{Symbol,Any}(
:(+) => +1.0,  # supports (N,) (A,) (N,N) (N,A) (A,N) (A,A); float arraymath abstractarraymath operators
:(-) => -1.0,  # supports (N,) (A,) (N,N) (N,A) (A,N) (A,A); float arraymath
:abs => :(sign(x)),             # float,operators
:ceil => 0,                     # float,operators
:float => 1, # float
:floor => 0, # float,operators
:isfinite => 0,                 # float,operators
:isinf => 0,                    # float,operators
:isnan => 0,                    # float,operators
:round => 0,                    # float,operators
:trunc => 0,                    # float,operators

)

float2arg = Dict{Symbol,Any}(
:+ => (1,1),                     # extra (N,) (A,); float,arraymath,abstractarraymath,operators
:- => (1,-1),                    # extra (N,) (A,); float,arraymath
:rem => (1,:(-trunc(x1./x2))),   # Remainder from Euclidean division, return same sign as x, missing (A,A); float,arraymath
:% => (1,:(-trunc(x1./x2))),     # same as rem
:mod => (1,:(-floor(x1./x2))),   # Modulus after division, return same sign as y; float,arraymath
)

float2zerograd = Dict{Symbol,Any}(
:(==) => 0,                      # supports any pair of values; float,operators,abstractarray
:< => 0,                         # only supports (N,N), arrays not supported; float,operators
:<= => 0,                        # only supports (N,N), arrays not supported; float,operators
:> => 0,                         # only supports (N,N), arrays not supported; operators
:>= => 0,                        # only supports (N,N), arrays not supported; operators
)
