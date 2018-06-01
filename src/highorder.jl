"""
    jacobian(f)

Computes the Jacobian of the vector valued function `f`. 
For `n = length(x)`, `m = length(f(x))`, the Jacobian
matrix  `jacobian(f)(x)` has dimension `m x n`.
"""
function jacobian(f)
    x -> begin
        m = length(f(x))
        hcat([grad(x->f(x)[i])(x) for i=1:m]...)'
    end
end

"""
    vjp(f)

Returns the  vector-Jacobian product operator
for the vector valued function `f`: 

    vjp(f)(x, v) == v' * jacobian(f)(x) 

This is more efficient than computing the full Jacobian
if only a few products are needed. See also `jvp`.
"""
function vjp(f)
    (x, v) -> begin
        grad(x -> v'f(x))(x)'
    end
end



# ref https://j-towns.github.io/2017/06/12/A-new-trick.html
"""
    jvp(f)

Returns the  Jacobian-vector product operator
for the vector valued function `f`: 

    jvp(f)(x, u) == jacobian(f)(x) * u
    
This is more efficient than computing the full Jacobian
if only a few products are needed. See also `vjp`.
"""
function jvp(f)
    (x, u) -> begin
        v = f(x) # any value of v should do
        g = v -> vjp(f)(x, v)'
        vjp(g)(v, u)'        
    end
end

"""
    hessian(f)

Compute the hessian of the scalar valued function `f`. 
It is equivalent to `jacobian(grad(f))`.
"""
hessian(f) = jacobian(grad(f))

"""
    vhp(f)

Returns the vector-Hessian product operator.
We have the equivalence `vhp(f)(x, v) == vjp(grad(f))(x, v)`.
"""
vhp(f) = vjp(x->vec(grad(f)(x)))

"""
    hvp(f)

Returns the Hessian product operat.
It is equivalent to `jvp(jacobian(f))` 
"""
hvp(f) = (x,v) -> vhp(f)(x,v)'   # can use vjp instead of jvp since the 
                        # hessian is symmetric