module MNIST2D
using AutoGrad
using GZip

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

function train(hidden...; lr=.1, epochs=20, seed=nothing)
    w = weights(hidden...; seed=seed)
    isdefined(:dtrn) || loaddata()
    println((0, loss(w,xtrn,ytrn), loss(w,xtst,ytst), accuracy(w,xtrn,ytrn), accuracy(w,xtst,ytst)))
    gradfun = grad(loss)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = gradfun(w, x, y)
            for i in 1:length(w)
                w[i] -= lr * g[i]
            end
        end
        println((epoch, loss(w,xtrn,ytrn), loss(w,xtst,ytst), accuracy(w,xtrn,ytrn), accuracy(w,xtst,ytst)))
    end
    return w
end

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

function loaddata()
    global xtrn, xtst, ytrn, ytst, dtrn
    xshape(a)=reshape(a./255f0,784,div(length(a),784))
    yshape(a)=(a[a.==0]=10; full(sparse(convert(Vector{Int},a),1:length(a),1f0)))
    xtrn = xshape(gzread("train-images-idx3-ubyte.gz")[17:end])
    xtst = xshape(gzread("t10k-images-idx3-ubyte.gz")[17:end])
    ytrn = yshape(gzread("train-labels-idx1-ubyte.gz")[9:end])
    ytst = yshape(gzread("t10k-labels-idx1-ubyte.gz")[9:end])
    dtrn = minibatch(xtrn, ytrn, 100)
end

function gzread(file; dir=Pkg.dir("AutoGrad/data/"), url="http://yann.lecun.com/exdb/mnist/")
    path = dir*file
    isfile(path) || download(url*file, path)
    f = gzopen(path)
    a = readbytes(f)
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

end # module
