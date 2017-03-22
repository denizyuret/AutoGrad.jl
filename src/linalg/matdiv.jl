# All of the reverse-mode sensitivities for operations of the form x1 \ x2.
ldiv2arg = [
    (:\,          :(-A_mul_Bt(At_ldiv_B(x1, dy), y)), :(At_ldiv_B(x1, dy))),
    (:At_ldiv_B,  :(-A_mul_Bt(y, x1 \ dy)),           :(x1 \ dy)),
    (:A_ldiv_Bt,  :(-A_mul_Bt(At_ldiv_B(x1, dy), y)), :(At_rdiv_B(dy, x1))),
    (:At_ldiv_Bt, :(-A_mul_Bt(y, x1 \ dy)),           :(At_rdiv_Bt(dy, x1))),
    (:Ac_ldiv_B,  :(-A_mul_Bc(y, x1 \ dy)),           :(x1 \ dy)),
    (:A_ldiv_Bc,  :(-A_mul_Bc(Ac_ldiv_B(x1, dy), y)), :(Ac_rdiv_B(dy, x1))),
    (:Ac_ldiv_Bc, :(-A_mul_Bc(y, x1 \ dy)),           :(Ac_rdiv_Bc(dy, x1))),
]

# All of the reverse-mode sensitivities for operations of the form x1 / x2.
rdiv2arg = [
    (:/,          :(A_rdiv_Bt(dy, x2)),  :(-At_mul_B(y, A_rdiv_Bt(dy, x2)))),
    (:At_rdiv_B,  :(A_ldiv_Bt(x2, dy)),  :(-At_mul_B(y, A_rdiv_Bt(dy, x2)))),
    (:A_rdiv_Bt,  :(dy / x2),            :(-At_ldiv_Bt(x2, dy) * y)),
    (:At_rdiv_Bt, :(At_ldiv_Bt(x2, dy)), :(-At_ldiv_Bt(x2, dy) * y)),
    (:Ac_rdiv_B,  :(A_ldiv_Bc(x2, dy)),  :(-Ac_mul_B(y, A_rdiv_Bc(dy, x2)))),
    (:A_rdiv_Bc,  :(dy / x2),            :(-Ac_ldiv_Bc(x2, dy) * y)),
    (:Ac_rdiv_Bc, :(Ac_ldiv_Bc(x2, dy)), :(-Ac_ldiv_Bc(x2, dy) * y)),
]

# Some matrices for unit testing.
A = randn(2, 2)
Σ = A'A + UniformScaling(1e-2)
X = randn(2, 2)

# Primitive definition and unit testing for ldiv operations.
for (f, g1, g2) in ldiv2arg
    @eval @primitive $f(x1, x2), dy, y $g1 $g2
    addtest(f, Σ, X)
end

# Primitive definition and unit testing for rdiv operations.
for (f, g1, g2) in rdiv2arg
    @eval @primitive $f(x1, x2), dy, y $g1 $g2
    addtest(f, X, Σ)
end
