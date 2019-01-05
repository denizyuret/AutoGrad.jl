include("header.jl")
using Statistics

@testset "core" begin

    # Issue #71.1: AutoGrad error in backprop when iterating dict
    u = [rand(2,3), rand(2)]
    v = [rand(1,2), rand(1)]
    m = Dict(:u=>u, :v=>v)
    x,y = rand(3,4),rand(1,4)
    pred(m,x) = foldl((x,w)->w[1]*x .+ w[2], [m[:u],m[:v]], init=x)
    loss(m,x,y) = mean(abs2, pred(m,x)-y)
    ∇ = grad(loss)
    @test isa(loss(m,x,y), Real)
    @test isa(∇(m,x,y), Dict)
    l2(ws) = mean(mean.(abs2, ws))
    loss(m,x,y) = mean(abs2, pred(m,x)-y) + mean(l2.(collect(values(m))))
    ∇ = grad(loss)
    @test isa(loss(m,x,y), Real)
    @test isa(∇(m,x,y), Dict)

    # Issue #71.2
    m = [rand(1,3), rand(1)]
    x,y = rand(3,4),rand(1,4)
    pred(m,x) = m[1]*x .+ m[2]
    loss(m,x,y) = mean(abs2, pred(m,x) .- y)
    ∇ = grad(loss)
    @test isa(∇(m,x,y), Array)
    loss(m,x,y) = mean(abs2, pred(m,x) .- y) + mean(mean.(abs2, m))
    ∇ = grad(loss)
    @test isa(∇(m,x,y), Array)

    # differentiate vs @diff with expression arguments
    x = Param(1)
    @test grad((@diff sin(sqrt(x))),x) == cos(x)/2
    @test grad(AutoGrad.differentiate(sin,sqrt(x)),x) != cos(x)/2
    
    # Issue #101.1
    x = Param(1.0); f1(x)=x
    @test grad((@diff f1(x)), x) == 1

    # Issue #101.2
    x = Param([1.,2.]); f2(x)=1x
    @test grad((@diff sum(f2.(x))), x) == [1.0,1.0]

    # Issue #103
    f(x) = exp(x)
    @test grad(x->f(x))(1.) == exp(1.)
    @test grad(x -> sum(f.(x)))([1.]) == [ exp(1.) ]

    # Double broadcasting
    x = Param([1.,2.]); f3(x)=sin(x); f4(x)=sin.(x)
    @test grad((@diff sum(f3.(x))), x) == grad((@diff sum(f4.(x))), x) == grad((@diff sum(f4(x))), x)

    # array-scalar mul
    a = Param(rand(2,3)); s = Param(rand())
    @test @gcheck sum(a .* s)
    @test @gcheck sum(a * s)

    # @zerograd needs to handle Bcasted
    x = Param(rand(2,3))
    f(x) = sign(x)
    @test @gcheck sum(f.(x))

    # result may not always be last on tape
    x = Param(rand(2,3))
    f(x) = (x1=sum(x); x2=2x; x1)
    @test @gcheck f(x)

    # Issue #106: 
    h(x) = exp(-x); h′(x,y) = -y
    𝓁(x,y) = sum(abs2,x-y)/2
    function neural_net(mparams, input; h=h, h′=h′, N=length(mparams))
        δ = [];
        X = Any[input];
        for i=1:N
            x = sum(mparams[i] .* [X[i],1])
            y = h.(x)
            push!(δ, h′.(x,y))
            push!(X,y)
        end
        return X,δ
    end
    mparams =[[randn(),randn()] for i=1:3]
    P = Param(mparams)
    loss(P,x,y)= 𝓁(neural_net(P,x)[1][end],y)
    x,y=randn(),randn()
    J = @diff loss(P,x,y)
    @test isa(J, AutoGrad.Tape)
    @test_broken @gcheck loss(P,x,y)

end
