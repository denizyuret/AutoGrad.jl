#  git	c1ed2e+	abb2cfe	e53b6d4
#  with	awsgpu	cpu	gpu	cpu	af	kn	kn+gc1	kn+gc2	kn+gc3	
#1 mul	0.67	0.95	0.56    0.94	0.56	0.56	0.56	0.56	0.56	
#2 bias	0.71	1.05	0.59    1.05	0.56	0.59	0.59	0.59	0.59	
#3 max	0.75	1.34	0.62    1.34	0.56	0.63	0.62	0.62	0.62	
#4 mul	0.81	1.43	0.75    1.44	0.74	0.75	0.75	0.75	0.75	
#5 bias	0.85	1.48	0.78    1.48	0.75	0.79	0.78	0.78	0.78	
#6 sub	0.89	1.49	0.82    1.49	0.81	0.82	0.81	0.81	0.82	
#7 sq	0.92	1.62	0.85    1.62	0.93	0.85	0.84	0.84	0.85	
#8 sum	1.21	1.63	1.06±01 1.62	1.22	1.19	1.07	1.08	1.07	
#9 forw	1.51	1.96±04	1.18±02 2.47	2.60	2.25	1.67	1.46	1.68	
#A grad	2.89	4.40±12	2.10±22	5.52	6.53	5.86	3.52	3.62	3.30	
#
# (*) timeall(weights(), weights(64), data(), 10)
# (*) af results with gc_enable=false and sync()
# (*) kn uses `similar`, +gc1 runs tmpfree every epoch, +gc2 runs tmpfree every iteration (minibatch), +gc3 uses KnetArray.
# AF: The forw records arrays preventing their reuse?
# AF: They are merging consecutive ops in one kernel, which breaks down with forw?

using AutoGrad, GZip, Compat
using AutoGrad: forward_pass

fun = []

push!(fun,(w,x,y)->w[1]*x)
push!(fun,(w,x,y)->w[1]*x.+w[2])
push!(fun,(w,x,y)->max(0,w[1]*x.+w[2]))
push!(fun,(w,x,y)->w[3]*max(0,w[1]*x.+w[2]))
push!(fun,(w,x,y)->w[3]*max(0,w[1]*x.+w[2]).+w[4])
push!(fun,(w,x,y)->((w[3]*max(0,w[1]*x.+w[2]).+w[4])-y))
push!(fun,(w,x,y)->(((w[3]*max(0,w[1]*x.+w[2]).+w[4])-y).^2))
fun1 = (w,x,y)->sum(((w[3]*max(0,w[1]*x.+w[2]).+w[4])-y).^2)
push!(fun, fun1)
push!(fun,(w,x,y)->forward_pass(fun1,(w,x,y),(),1))
push!(fun,grad(fun1))

function timeall(w=w2,d=d0,t=10)
    for i=1:length(fun)
        printfun(fun[i])
        for j=1:3
            sleep(2)
            @time loop(fun[i],w,d,t)
        end
    end
end

function loop(f,w,d,t)
    for i in 1:t
        for (x,y) in d
            f(w,x,y)
        end
    end
end

function weights(h...; seed=nothing)
    seed==nothing || srand(seed)
    w = Array{Float32}[]
    x = 28*28
    for y in [h..., 10]
        push!(w, convert(Array{Float32}, 0.1*randn(y,x)))
        push!(w, zeros(Float32,y))
        x = y
    end
    return w
end

function data()
    info("Loading data...")
    xshape(a)=reshape(a./255f0,784,div(length(a),784))
    yshape(a)=(a[a.==0]=10; full(sparse(convert(Vector{Int},a),1:length(a),1f0)))
    xtrn = xshape(gzload("train-images-idx3-ubyte.gz")[17:end])
    ytrn = yshape(gzload("train-labels-idx1-ubyte.gz")[9:end])
    #xtst = xshape(gzload("t10k-images-idx3-ubyte.gz")[17:end])
    #ytst = yshape(gzload("t10k-labels-idx1-ubyte.gz")[9:end])
    batch(xtrn,ytrn,100)
end

function gzload(file; path=joinpath(AutoGrad.datapath,file), url="http://yann.lecun.com/exdb/mnist/$file")
    isfile(path) || download(url, path)
    f = gzopen(path)
    a = @compat read(f)
    close(f)
    return(a)
end

function batch(x, y, batchsize)
    data = Any[]
    nx = size(x,2)
    for i=1:batchsize:nx
        j=min(i+batchsize-1,nx)
        push!(data, (x[:,i:j], y[:,i:j]))
    end
    return data
end

function printfun(x)
    if isdefined(x,:code)
        println(Base.uncompressed_ast(x.code).args[3].args[2].args[1])
    else
        println(x)
    end
end

if !isdefined(:d0)
    d0 = data()
    w1 = weights(seed=1)
    w2 = weights(64;seed=1)
end

:ok

# julia> timeall()
# (Main.getindex)(w,1) * x
#   0.956369 seconds (30.00 k allocations: 147.400 MB, 0.97% gc time)
#   0.947161 seconds (30.00 k allocations: 147.400 MB, 0.78% gc time)
#   0.947129 seconds (30.00 k allocations: 147.400 MB, 0.66% gc time)
# (Main.getindex)(w,1) * x .+ (Main.getindex)(w,2)
#   1.055720 seconds (144.00 k allocations: 297.913 MB, 1.41% gc time)
#   1.054730 seconds (144.00 k allocations: 297.913 MB, 1.41% gc time)
#   1.054276 seconds (144.00 k allocations: 297.913 MB, 1.31% gc time)
# (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2))
#   1.356736 seconds (168.03 k allocations: 445.315 MB, 1.63% gc time)
#   1.353720 seconds (168.00 k allocations: 445.313 MB, 1.55% gc time)
#   1.353312 seconds (168.00 k allocations: 445.313 MB, 1.56% gc time)
# (Main.getindex)(w,3) * (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2))
#   1.443865 seconds (180.04 k allocations: 468.844 MB, 1.62% gc time)
#   1.440977 seconds (180.00 k allocations: 468.842 MB, 1.56% gc time)
#   1.441619 seconds (180.00 k allocations: 468.842 MB, 1.63% gc time)
# (Main.getindex)(w,3) * (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2)) .+ (Main.getindex)(w,4)
#   1.486593 seconds (288.04 k allocations: 495.852 MB, 1.63% gc time)
#   1.487364 seconds (288.00 k allocations: 495.850 MB, 1.69% gc time)
#   1.485600 seconds (288.00 k allocations: 495.850 MB, 1.69% gc time)
# ((Main.getindex)(w,3) * (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2)) .+ (Main.getindex)(w,4)) - y
#   1.498100 seconds (300.04 k allocations: 519.382 MB, 1.70% gc time)
#   1.498842 seconds (300.00 k allocations: 519.379 MB, 1.78% gc time)
#   1.497004 seconds (300.00 k allocations: 519.379 MB, 1.78% gc time)
# (((Main.getindex)(w,3) * (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2)) .+ (Main.getindex)(w,4)) - y) .^ 2
#   1.634301 seconds (318.04 k allocations: 543.277 MB, 1.73% gc time)
#   1.631012 seconds (318.00 k allocations: 543.274 MB, 1.66% gc time)
#   1.632553 seconds (318.00 k allocations: 543.274 MB, 1.96% gc time)
# (Main.sum)((((Main.getindex)(w,3) * (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2)) .+ (Main.getindex)(w,4)) - y) .^ 2)
#   1.636876 seconds (324.00 k allocations: 543.365 MB, 1.74% gc time)
#   1.635672 seconds (324.00 k allocations: 543.365 MB, 1.74% gc time)
#   1.636505 seconds (324.00 k allocations: 543.365 MB, 1.74% gc time)
# (Main.forward_pass)(Main.fun1,(top(tuple))(w,x,y),(top(tuple))(),1)
#   2.109984 seconds (1.45 M allocations: 601.044 MB, 1.89% gc time)
#   2.110840 seconds (1.45 M allocations: 601.044 MB, 1.91% gc time)
#   2.109753 seconds (1.45 M allocations: 601.044 MB, 1.83% gc time)
# gradfun
#   4.794647 seconds (3.39 M allocations: 2.200 GB, 2.58% gc time)
#   4.788467 seconds (3.39 M allocations: 2.200 GB, 2.54% gc time)
#   4.790467 seconds (3.39 M allocations: 2.200 GB, 2.57% gc time)


# julia> include(Pkg.dir("Knet/test/profile_kn.jl"))
# before d0kn,w2kn (4934356992,:cuda_ptrs,0)
# after d0kn,w2kn (4720431104,(4000,600,0),(200704,1,0),(2560,1,0),(40,1,0),(256,1,0),(313600,600,0),:cuda_ptrs,0)
# 1(Main.getindex)(w,1) * x
#   0.615343 seconds (278.52 k allocations: 10.871 MB)
#   0.559348 seconds (222.00 k allocations: 8.331 MB)
#   0.559312 seconds (222.00 k allocations: 8.331 MB)
# 2(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2)
#   1.278285 seconds (374.67 k allocations: 14.424 MB)
#   0.593602 seconds (330.00 k allocations: 12.268 MB)
#   0.592977 seconds (312.00 k allocations: 11.627 MB)
# 3(Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2))
#   0.632009 seconds (371.78 k allocations: 13.691 MB)
#   0.623921 seconds (348.00 k allocations: 12.817 MB)
#   0.624452 seconds (366.00 k allocations: 13.458 MB)
# 4(Main.getindex)(w,3) * (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2))
#   0.755348 seconds (528.04 k allocations: 20.327 MB)
#   0.752531 seconds (510.00 k allocations: 19.684 MB)
#   0.752733 seconds (528.00 k allocations: 20.325 MB)
# 5(Main.getindex)(w,3) * (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2)) .+ (Main.getindex)(w,4)
#   0.787278 seconds (618.04 k allocations: 23.623 MB)
#   0.784750 seconds (618.00 k allocations: 23.621 MB)
#   0.785054 seconds (618.00 k allocations: 23.621 MB)
# 6((Main.getindex)(w,3) * (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2)) .+ (Main.getindex)(w,4)) - y
#   0.852492 seconds (705.88 k allocations: 27.000 MB)
#   0.815621 seconds (654.00 k allocations: 24.811 MB)
#   0.815684 seconds (654.00 k allocations: 24.811 MB)
# 7(((Main.getindex)(w,3) * (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2)) .+ (Main.getindex)(w,4)) - y) .^ 2
#   0.853973 seconds (713.80 k allocations: 26.875 MB)
#   0.845969 seconds (690.00 k allocations: 26.001 MB)
#   0.846093 seconds (690.00 k allocations: 26.001 MB)
# 8(Main.sum)((((Main.getindex)(w,3) * (Main.max)(0,(Main.getindex)(w,1) * x .+ (Main.getindex)(w,2)) .+ (Main.getindex)(w,4)) - y) .^ 2)
#   1.065069 seconds (697.55 k allocations: 26.163 MB)
#   1.059029 seconds (696.00 k allocations: 26.093 MB)
#   1.058951 seconds (696.00 k allocations: 26.093 MB)
# 9(Main.forward_pass)(Main.fun1,(top(tuple))(w,x,y),(top(tuple))(),1)
#   1.606405 seconds (2.22 M allocations: 101.994 MB)
#   1.270358 seconds (1.90 M allocations: 87.799 MB)
#   1.270771 seconds (1.90 M allocations: 87.799 MB)
# 10gradfun
#   4.155291 seconds (4.51 M allocations: 188.382 MB)
#   2.494650 seconds (4.09 M allocations: 171.112 MB)
#   2.528842 seconds (4.09 M allocations: 171.112 MB)
