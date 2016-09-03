# work in progress...

bessel1arg = [
(airyai, :(airyaiprime(x))),
(airyaiprime, :(x.*airyai(x))),
(airybi, :(airybiprime(x))),
(airybiprime, :(x.*airybi(x))),
(airyprime, :(x.*airyai(x))),
(besselj0, :(-besselj1(x))),
(besselj1, :((besselj0(x)-besselj(2,x))/2)),
(bessely0, :(-bessely1(x))),
(bessely1, :((bessely0(x)-bessely(2,x))/2)),
]

for (f,g) in bessel1arg
    @eval @primitive $f(x),dy,y  (dy.*($g))
    addtest1(f,(-Inf,Inf))
end

@primitive besselj(nu,x)  error("No gradient for besselj")  error("No gradient for besselj")
@primitive bessely(nu,x)  error("No gradient for besselj")  error("No gradient for besselj")

# The following defined for (nu,x) where nu can be any integer or
# positive real

# bessel2arg = [
# (besseli, :((besseli(nu-1,x)+besseli(nu+1,x))/2)),
# (besselix, :((besselix(nu-1,x)+besselix(nu+1,x))/2)),
# (besselj, :((besselj(nu-1,x)-besselj(nu+1,x))/2)),
# (besseljx, :((besseljx(nu-1,x)-besseljx(nu+1,x))/2)),
# #besselk,
# #besselkx,
# #bessely,
# #besselyx,
# ]

# for (f,g) in bessel2arg
#     @eval @primitive $f(nu,x),dy,y  nothing  unbroadcast(x, dy.*($g))
#     addtest2(f,(0,Inf),(-Inf,Inf))
# end

# bessel2arg2 = [
# airy,
# airyx,
# # besselh  # has (nu,k,z) and (nu,z) argtypes where k=int
# # besselhx # (nu,k,x), no vectorization
# # hankelh1 # (nu,x), vectorized
# # hankelh1x # (nu,x), vectorized
# # hankelh2
# # hankelh2x
# ]
# for f in bessel2arg2
#     @eval @zerograd $f(x1,x2)
# end

