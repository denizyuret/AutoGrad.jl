"""
This example uses the Housing dataset from the UCI Machine Learning
Repository to demonstrate a linear regression model. The dataset has
housing related information for 506 neighborhoods in Boston from
1978. Each neighborhood has 14 attributes, the goal is to use the
first 13, such as average number of rooms per house, or distance to
employment centers, to predict the 14’th attribute: median dollar
value of the houses.

To run the demo, simply `include("housing.jl")` and run `Housing.train()`.  
The dataset will be automatically downloaded.  You can provide the
initial weights as an optional argument, which should be a pair of
1x13 weight matrix and a scalar bias.  `train` also accepts the
following keyword arguments: `lr` specifies the learning rate,
`epochs` gives number of epochs, and `seed` specifies the random
number seed.  The quadratic loss for the train and test sets will be
printed at every epoch and optimized parameters will be returned.
"""
module Housing
using AutoGrad
using Main

function loss(wb, x=xtrn, ygold=ytrn)
    (w,b) = wb
    ypred = w*x .+ b
    sum(abs2,ypred - ygold) / size(ygold,2)
end

function train(w=Any[0.1*randn(1,13), 0.0]; lr=.1, epochs=20, seed=nothing)
    seed==nothing || (srand(seed); loaddata())
    isdefined(Housing,:xtrn) || loaddata()
    gradfun = grad(loss)
    println((0, :trnloss, loss(w,xtrn,ytrn), :tstloss, loss(w,xtst,ytst)))
    for epoch=1:epochs
        g = gradfun(w, xtrn, ytrn)
        for i in 1:length(w)
            w[i] -= lr * g[i]
        end
        println((epoch, :trnloss, loss(w,xtrn,ytrn), :tstloss, loss(w,xtst,ytst)))
    end
    return w
end

function loaddata()
    global xtrn, ytrn, xtst, ytst
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    file=AutoGrad.dir("data", "housing.data")
    if !isfile(file)
        info("Downloading $url to $file")
        download(url, file)
    end
    data = readdlm(file)'
    @show size(data) # (14,506)
    x = data[1:13,:]
    y = data[14:14,:]
    x = (x .- mean(x,2)) ./ std(x,2) # Data normalization
    r = randperm(size(x,2))          # trn/tst split
    xtrn=x[:,r[1:400]]
    ytrn=y[:,r[1:400]]
    xtst=x[:,r[401:end]]
    ytst=y[:,r[401:end]]
end

end # module Housing
