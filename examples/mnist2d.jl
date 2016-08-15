isdefined(:MNIST) || include(Pkg.dir("Knet/examples/mnist.jl"))

#module MNIST2D
#using AutoGrad: dbg, grad
#using Main: MNIST
# TODO: get this independent of Knet at some point

function test()
    loaddata()
    dbg(:core)
    global w = weights(64)
    global g = grad(loss)
    # TODO: This is too slow, need quick version:
    # check_grads(loss, w)
end

# TODO: take number of layers as an argument? no actually we can just pass the parameters.  have a utility function that computes parameters for a given shape.
# TODO: check grads, and check loss with Knet.

function train(w=weights(); lr=.1, epochs=20)
    isdefined(:dtrn) || loaddata()
    gradfun = grad(loss)
    println((0, loss(w,xtrn,ytrn), loss(w,xtst,ytst)))
    for epoch=1:epochs
        for (x,y) in dtrn
            g = gradfun(w, x, y)
            for i in 1:length(w)
                w[i] -= lr * g[i]
            end
        end
        println((epoch, loss(w,xtrn,ytrn), loss(w,xtst,ytst)))
    end
    return w
end

function loss(w, x=xtst, ygold=ytst)
    i = 1
    while i+2 < length(w)
        x = max(0, w[i]*x .+ w[i+1])
        i += 2
    end
    ypred = w[i]*x .+ w[i+1]
    ynorm = ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

function loaddata()
    global xtrn, xtst, ytrn, ytst, dtrn
    xtrn = reshape2(MNIST.xtrn)
    xtst = reshape2(MNIST.xtst)
    ytrn = MNIST.ytrn
    ytst = MNIST.ytst
    dtrn = minibatch(xtrn, ytrn, 100)
end

function minibatch(x, y, batchsize)
    data = Any[]
    nx = size(x,2)
    for i=1:batchsize:nx
        j=min(i+batchsize-1,nx)
        push!(data, (x[:,i:j], y[:,i:j]))
    end
    return data
end

reshape2(a)=reshape(a,div(length(a),size(a,ndims(a))),size(a,ndims(a)))

function weights(h...; seed=nothing)
    seed==nothing || srand(seed)
    w = Any[]
    x = 28*28
    for y in [h..., 10]
        push!(w, 0.1*randn(y,x))
        push!(w, zeros(y))
        x = y
    end
    return w
end

#end # module
