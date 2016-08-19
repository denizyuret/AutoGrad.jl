# Adapted from mnist.jl

using AutoGrad
using GZip
using Main
using Compat

function predict(w, x)
    i = 1
    while i+2 < length(w)
        x = max(0, w[i]*x .+ w[i+1])
        i += 2
    end
    return w[i]*x .+ w[i+1]
end

function loss(w, x, ygold)
    ypred = predict(w, x)
    ynorm = ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

function accuracy(w, x, ygold)
    ypred = predict(w, x)
    sum((ypred .== maximum(ypred,1)) & (ygold .== maximum(ygold,1))) / size(ygold,2)
end

function profit(f; epochs=10)
    timeit(f; epochs=epochs)
    w = weights(64; seed=1);
    sleep(2)
    gc_enable(false)
    @profile f(w; epochs=epochs)
    gc_enable(true)
end

function timeit(f; epochs=1)
    isdefined(:dtrn) || loaddata()
    for i=1:3
        w = weights(64; seed=1);
        sleep(2)
        gc_enable(false)
        @time f(w; epochs=epochs)
        gc_enable(true)
    end
end

function train0(w=weights(64); lr=.1, epochs=1)
    gradfun = grad(loss)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = gradfun(w, x, y)
            for i in 1:length(w)
                # w[i] -= lr * g[i]
                Base.axpy!(-lr, g[i], w[i])
            end
        end
    end
    return w
end

function train1(w=weights(64); lr=.1, epochs=1)
    gradfun = grad(loss)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = gradfun(w, x, y)
        end
    end
    return w
end

function train2(w=weights(64); lr=.1, epochs=1)
    for epoch=1:epochs
        for (x,y) in dtrn
            z = loss(w, x, y)
        end
    end
    return w
end

function train3(w=weights(64); lr=.1, epochs=1)
    for epoch=1:epochs
        for (x,y) in dtrn
            y = predict(w, x)
        end
    end
    return w
end

function train4(w=weights(64); lr=.1, epochs=1)
    for epoch=1:epochs
        for (x,y) in dtrn
            z = AutoGrad.forward_pass(loss, (w, x, y), (), 1)
        end
    end
    return w
end

function weights(h...; seed=nothing)
    seed==nothing || srand(seed)
    w = Any[]
    x = 28*28
    for y in [h..., 10]
        push!(w, convert(Array{Float32}, 0.1*randn(y,x)))
        push!(w, zeros(Float32,y))
        x = y
    end
    return w
end

function loaddata()
    info("Loading data...")
    global xtrn, xtst, ytrn, ytst, dtrn
    xshape(a)=reshape(a./255f0,784,div(length(a),784))
    yshape(a)=(a[a.==0]=10; full(sparse(convert(Vector{Int},a),1:length(a),1f0)))
    xtrn = xshape(gzread("train-images-idx3-ubyte.gz")[17:end])
    xtst = xshape(gzread("t10k-images-idx3-ubyte.gz")[17:end])
    ytrn = yshape(gzread("train-labels-idx1-ubyte.gz")[9:end])
    ytst = yshape(gzread("t10k-labels-idx1-ubyte.gz")[9:end])
    dtrn = minibatch(xtrn, ytrn, 100)
    info("Loading done...")
end

function gzread(file; dir=Pkg.dir("AutoGrad/data/"), url="http://yann.lecun.com/exdb/mnist/")
    path = dir*file
    isfile(path) || download(url*file, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
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
