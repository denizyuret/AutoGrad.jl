module Housing
using AutoGrad
using Main

function loss(wb, x=xtrn, ygold=ytrn)
    (w,b) = wb
    ypred = w*x .+ b
    sum(abs2(ypred - ygold)) / size(ygold,2)
end

function train(w=Any[0.1*randn(1,13), 0.0]; lr=.1, epochs=20, seed=nothing)
    seed==nothing || (srand(seed); loaddata())
    isdefined(Housing,:xtrn) || loaddata()
    gradfun = grad(loss)
    println((0, loss(w,xtrn,ytrn), loss(w,xtst,ytst)))
    for epoch=1:epochs
        g = gradfun(w, xtrn, ytrn)
        for i in 1:length(w)
            w[i] -= lr * g[i]
        end
        println((epoch, loss(w,xtrn,ytrn), loss(w,xtst,ytst)))
    end
    return w
end

function loaddata()
    global xtrn, ytrn, xtst, ytst
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    file=Pkg.dir("AutoGrad/data/housing.data")
    if !isfile(file)
        info("Downloading $url to $file")
        download(url, file)
    end
    data = readdlm(file)'
    @show size(data) # (14,506)
    x = data[1:13,:]
    y = data[14,:]
    x = (x .- mean(x,2)) ./ std(x,2) # Data normalization
    r = randperm(size(x,2))          # trn/tst split
    xtrn=x[:,r[1:400]]
    ytrn=y[:,r[1:400]]
    xtst=x[:,r[401:end]]
    ytst=y[:,r[401:end]]
end

end # module Housing
