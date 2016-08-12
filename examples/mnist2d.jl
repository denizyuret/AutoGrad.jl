using Knet

isdefined(:MNIST) || include(Pkg.dir("Knet/examples/mnist.jl"))

training_batches = minibatch(MNIST.xtrn, MNIST.ytrn, 100)

mnist_softmax(w, x, y)=cross_entropy(y, softmax(w[1]*x .+ w[2]))

function softmax(x, y=similar(x))
    (st,nx) = size2(x)
    @inbounds for j=1:nx
        i1=(j-1)*st+1
        i2=j*st
        xmax = typemin(eltype(x))
        ysum = zero(Float64)
        for i=i1:i2; x[i] > xmax && (xmax = x[i]); end
        for i=i1:i2; ysum += (y[i]=exp(x[i] - xmax)); end
        for i=i1:i2; y[i] /= ysum; end
    end
    return y
end
