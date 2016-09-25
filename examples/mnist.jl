"""
This example learns to classify hand-written digits from the MNIST
dataset.  There are 60000 training and 10000 test examples. Each input
x consists of 784 pixels representing a 28x28 image.  The pixel values
have been normalized to [0,1]. Each output y is a ten-dimensional
one-hot vector (a vector that has a single non-zero component)
indicating the correct class (0-9) for a given image.

To run the demo, simply `include("mnist.jl")` and run `MNIST.train()`.
The dataset will be automatically downloaded.  You can provide the
initial weights as an optional argument to `train`, which should have
the form [w0,b0,w1,b1,...] where wi (with size = output x input) is
the weight matrix and bi (with size = output) is the bias vector for
layer i.  The function `MNIST.weights(h...)` can be used to create
random starting weights for a neural network with hidden sizes (h...).
If not specified, default weights are created using `MNIST.weights()`
which correspond to a 0 hidden layer network, i.e. a softmax model.
`train` also accepts the following keyword arguments: `lr` specifies
the learning rate, `epochs` gives number of epochs.  The cross entropy
loss and accuracy for the train and test sets will be printed at every
epoch and optimized parameters will be returned.
"""
module MNIST
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

function train(w=weights(); lr=.1, epochs=20)
    isdefined(MNIST,:dtrn) || loaddata()
    println((0, :ltrn, loss(w,xtrn,ytrn), :ltst, loss(w,xtst,ytst), :atrn, accuracy(w,xtrn,ytrn), :atst, accuracy(w,xtst,ytst)))
    gradfun = grad(loss)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = gradfun(w, x, y)
            for i in 1:length(w)
                w[i] -= lr * g[i]
            end
        end
        println((epoch, :ltrn, loss(w,xtrn,ytrn), :ltst, loss(w,xtst,ytst), :atrn, accuracy(w,xtrn,ytrn), :atst, accuracy(w,xtst,ytst)))
    end
    return w
end

function weights(h...; seed=nothing)
    seed==nothing || srand(seed)
    w = Any[]
    x = 28*28
    for y in [h..., 10]
        push!(w, convert(Array{Float32}, 0.1*randn(y,x)))
        push!(w, zeros(Float32, y))
        x = y
    end
    return w
end

function loaddata()
    info("Loading data...")
    global xtrn, xtst, ytrn, ytst, dtrn
    xshape(a)=reshape(a./255f0,784,div(length(a),784))
    yshape(a)=(a[a.==0]=10; full(sparse(convert(Vector{Int},a),1:length(a),1f0)))
    xtrn = xshape(gzload("train-images-idx3-ubyte.gz")[17:end])
    xtst = xshape(gzload("t10k-images-idx3-ubyte.gz")[17:end])
    ytrn = yshape(gzload("train-labels-idx1-ubyte.gz")[9:end])
    ytst = yshape(gzload("t10k-labels-idx1-ubyte.gz")[9:end])
    dtrn = minibatch(xtrn, ytrn, 100)
    info("Loading done...")
end

function gzload(file; path=joinpath(AutoGrad.datapath,file), url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
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

end # module
