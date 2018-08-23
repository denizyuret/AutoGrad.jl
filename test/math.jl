include("header.jl")

@testset "math" begin
    o = (:delta=>0.0001,:atol=>0.01,:rtol=>0.01)
    ϵ = 0.1
    abs_gt_0(x)=sign(x) * (abs(x) + ϵ)
    abs_gt_1(x)=sign(x) * (abs(x) + 1 + ϵ)
    abs_lt_1(x)=rand() * (2-2ϵ) - (1-ϵ)
    val_lt_1(x)=rand() * (1-2ϵ) + ϵ
    val_gt_1(x)=abs(x) + 1 + ϵ
    val_gt_0(x)=abs(x) + ϵ
    val_gt_m1(x)=abs(x) - 1 + ϵ
    val_cot(x)=(abs(cot(x)) > 10 ? x+pi/2 : x) # avoid multiples of pi
    val_cotd(x)=(abs(cotd(x)) > 10 ? x+90 : x) # avoid multiples of 180
    val_tan(x)=(abs(tan(x)) > 10 ? x+pi/2 : x) # avoid pi/2, 3pi/2 etc.
    val_tand(x)=(abs(tand(x)) > 10 ? x+90 : x) # avoid 90, 270, etc.
    val_logb(x)=(x = abs(x) + ϵ; abs(x-1) < ϵ ? x+0.5 : x)     # avoid log base close to 1
    id = identity
    val(v) = (x->v)

    @test randcheck(acos,abs_lt_1;o...)
    @test randcheck(acosd,abs_lt_1;o...)
    @test randcheck(acosh,val_gt_1;o...)
    @test randcheck(acot,abs_gt_0;o...)
    @test randcheck(acotd,abs_gt_0;o...)
    @test randcheck(acoth,abs_gt_1;o...)
    @test randcheck(acsc,abs_gt_1;o...)
    @test randcheck(acscd,abs_gt_1;o...)
    @test randcheck(acsch,abs_gt_0;o...)
    @test randcheck(asec,abs_gt_1;o...)
    @test randcheck(asecd,abs_gt_1;o...)
    @test randcheck(asech,val_lt_1;o...)
    @test randcheck(asin,abs_lt_1;o...)
    @test randcheck(asind,abs_lt_1;o...)
    @test randcheck(asinh;o...)
    @test randcheck(atan;o...)
    @test randcheck(atand;o...)
    @test randcheck(atanh,abs_lt_1;o...)
    @test randcheck(cbrt;o...)
    @test randcheck(clamp,identity,val(-0.5),val(0.5); args=1,o...)
    #@test randcheck(clamp!) # overwrites
    @test randcheck(cos;o...)
    @test randcheck(cosc;o...)
    @test randcheck(cosd;o...)
    @test randcheck(cosh;o...)
    @test randcheck(cospi;o...)
    @test randcheck(cot,val_cot;o...)
    @test randcheck(cotd,val_cotd;o...)
    @test randcheck(coth,abs_gt_0;o...)
    @test randcheck(csc,val_cot;o...)
    @test randcheck(cscd,val_cotd;o...)
    @test randcheck(csch,abs_gt_0;o...)
    @test randcheck(deg2rad;o...)
    @test randcheck(exp;o...)
    @test randcheck(exp10,abs_lt_1;o...)
    @test randcheck(exp2;o...)
    @test randcheck(expm1;o...)
    @test randcheck(exponent;o...)
    #@test randcheck(frexp) # returns 2 values
    @test randcheck(hypot,id,id;o...)
    @test randcheck(ldexp,id,val(rand(-5:5)); args=1,o...)
    @test randcheck(log,val_gt_0;o...)
    @test randcheck(log,val_logb,val_gt_0;o...)
    @test randcheck(log10,val_gt_0;o...)
    @test randcheck(log1p,val_gt_m1;o...)
    @test randcheck(log2,val_gt_0;o...)
    @test randcheck(max,id,id;o...)
    @test randcheck(min,id,id;o...)
    #@test randcheck(minmax) # returns 2 values
    @test randcheck(mod2pi;o...)
    #@test randcheck(modf)   # returns 2 values
    @test randcheck(rad2deg;o...)
    @test randcheck(rem2pi,id,val(RoundNearest); args=1,o...)
    @test randcheck(sec,val_tan;o...)
    @test randcheck(secd,val_tand;o...)
    @test randcheck(sech;o...)
    @test randcheck(significand;o...)
    @test randcheck(sin;o...)
    @test randcheck(sinc;o...)
    #@test randcheck(sincos) # returns 2 values
    @test randcheck(sind;o...)
    @test randcheck(sinh;o...)
    @test randcheck(sinpi;o...)
    @test randcheck(sqrt,val_gt_0;o...)
    @test randcheck(tan, val_tan;o...)
    @test randcheck(tand, val_tand;o...)
    @test randcheck(tanh;o...)
end
