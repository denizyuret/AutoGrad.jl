trig1arg = Dict{Symbol,Any}(
:acosd => :(-(180/pi)./sqrt(1-abs2(x))),  # domain: abs(x) <= 1; trig,operators
:acot => :(-1./(1+abs2(x))),              # trig
:acotd => :(-(180/pi)./(1+abs2(x))),      # trig,operators
:acoth => :(1./(1-abs2(x))),              # domain: abs(x) >= 1; trig
:acsc => :(-1./sqrt(x.*x.*(x-1).*(x+1))), # domain: abs(x) >= 1; trig
:acscd => :(-(180/pi)./sqrt(x.*x.*(x-1).*(x+1))), # domain: abs(x) >= 1; trig,operators
:acsch => :(-1./sqrt(x.^4+x.^2)),         # trig
:asec => :(1./sqrt(x.^4-x.^2)),           # domain: abs(x) >= 1; trig
:asecd => :((180/pi)./sqrt(x.^4-x.^2)),   # domain: abs(x) >= 1; trig,operators
:asech => :(-1./sqrt(x.^2-x.^4)),         # domain: 0 < x <= 1; trig
#:asind => :todo,                         # domain: abs(x) <= 1; trig,operators
#:atand => :todo,                         # trig,operators
#:cosc => :todo,   # trig,operators
#:cosd => :todo,   # trig,operators
:cospi => :(-sinpi(x).*pi),  # trig,operators
#:cot => :todo,  # trig
#:cotd => :todo,  # trig
#:coth => :todo,  # trig
#:csc => :todo,   # trig
#:cscd => :todo,  # trig
#:csch => :todo,  # trig
#:sec => :todo,  # trig
#:secd => :todo, # trig
#:sech => :todo, # trig
#:sinc => :todo, # trig,operators
#:sind => :todo,  # trig,operators
:sinpi => :(cospi(x).*pi),      # trig,operators
#:tand => :todo,  # trig,operators
)

defgrads(trig1arg, Number)
defgrads(trig1arg, AbstractArray)

for (f,g) in ((:acosd, :cosd),
              (:acotd, :cotd),
              (:acscd, :cscd),
              (:asecd, :secd))
    gx = eval(g)
    fx = x->gx(180x/pi)
    testargs(::Fn{f},a...)=map(fx, testargs(Fn2(f),a...))
end

for (f,g) in ((:acot, :cot),
              (:acoth, :coth),
              (:acsc, :csc),
              (:acsch, :csch),
              (:asec, :sec),
              (:asech, :sech))
    gx = eval(g)
    testargs(::Fn{f},a...)=map(gx, testargs(Fn2(f),a...))
end
