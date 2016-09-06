"""
rnn_lonely_integer.jl: (c) Ozan Arkan Can, 2016

This example learns to find the lonely integer in an array where all
but one of the integers occur in pairs.  The problem is taken from
[hackerrank](https://www.hackerrank.com/challenges/lonely-integer).
We feed elements into a recurrent neural network one by one, and we
get a prediction from the model after the final element.

To run the demo, simply `include("rnn_lonely_integer.jl")` and run
`LonelyInteger.train()`.  You can provide the initial weights as an
optional argument to `train`, which should have the form
[Whx,Whh,bh,Woh,bo] where first three elements are the parameters of
the rnn and the last two are the parameters of the softmax classifier.
The function `LonelyInteger.weights(;h, vocab)` can be used to create
random starting weights for a recurrent neural network with hidden
size and vocab size.  `train` also accepts the following keyword
arguments: `lr` specifies the learning rate, `N` gives the number of
instances that are used to train the model.  `seqlength` specifies the
length of the input sequences and `limit` defines the range of the
elements (from 1 to limit).  The running average cross entropy loss
and accuracy for the seen data will be printed after every 10k
instances and optimized parameters will be returned.

Data instances are created using `gendata`. It generates one instance
for a given sequence length and the limit parameter.

`test_example` takes trained weights and optional sequence length and
limit (must be same as the limit parameter used in the training)
parameters.  It shows a generated sequence and prediction of the model
for that instance.  You can test the performance of the model on
shorter or longer sequences than sequences used in training.

You can see an example experiment log at the end of the file.
"""

module LonelyInteger

using AutoGrad

function rnn_sequence(w, X; hidden=256)
    h = zeros(Float32, hidden, 1)

    for x in X
	preh = w[1] * x + w[2] * h .+ w[3]
	h = tanh(preh)
    end

    return h
end

function gendata(;seqlength=5, limit=20)
    rnums = randperm(limit)
    seq = rnums[1:(seqlength-1)]
    append!(seq, rnums[1:(seqlength-1)])
    push!(seq, rnums[seqlength])

    function onehot(indx)
	rep = zeros(Float32, limit, 1)
	rep[indx, 1] = 1.0
	return rep
    end

    onehotseq = map(onehot, seq)
    y = onehotseq[end]
    shuffle!(onehotseq)
    return (onehotseq, y)
end

function predict(w, h)
    return w[1] * h .+ w[2]
end

function loss(w, X, ygold)
    hT = rnn_sequence(w[1:3], X)
    ypred = predict(w[4:5], hT)
    ynorm = ypred .- log(sum(exp(ypred),1))
    -sum(ygold .* ynorm) / size(ygold,2)
end

function accuracy(w, X, ygold)
    hT = rnn_sequence(w[1:3], X)
    ypred = predict(w[4:5], hT)
    acc = indmax(ypred) == indmax(ygold) ? 1 : 0
end

function train(; lr=.001, N=2000000, seqlength=7, limit=50, w=weights(;vocab=limit))
    gradfun = grad(loss)
    nextn = 1000

    lss_avg = 0.0
    acc_avg = 0.0

    for n=1:N
	seq, ygold = gendata(;seqlength=seqlength, limit=limit)
	g = gradfun(w, seq, ygold)
	
	sloss = loss(w, seq, ygold)
	acc = accuracy(w, seq, ygold)

	#update
	for i=1:length(w); w[i] -= lr * g[i]; end
	
	lss_avg = (n==1 ? sloss : 0.99 * lss_avg + 0.01 * sloss)
	acc_avg = (n==1 ? acc : 0.99 * acc_avg + 0.01 * acc)
	
	if acc_avg > 0.999
	    println((n, lss_avg, acc_avg))
	    break
	end

	(n == nextn) && (println((n,lss_avg, acc_avg)); nextn+=1000)
    end

    return w
end

function timing(; lr=.001, N=10, seqlength=15, limit=100, w=weights(;vocab=limit))
    gradfun = grad(loss)

    function onestep()
	seq, ygold = gendata(;seqlength=seqlength, limit=limit);
	g = gradfun(w, seq, ygold);
	
	#update
	for i=1:length(w); w[i] -= lr * g[i]; end
    end

    for n=1:N
	gc_enable(false)
	@time onestep()
	gc_enable(true)
    end
end


function gendata(;seqlength=5, limit=20)
    rnums = randperm(limit)
    dup = round(Int32, (seqlength - 1) / 2)
    seq = rnums[1:dup]
    append!(seq, rnums[1:dup])
    push!(seq, rnums[dup+1])

    function onehot(indx)
	rep = zeros(Float32, limit, 1)
	rep[indx, 1] = 1.0
	return rep
    end

    onehotseq = map(onehot, seq)
    y = onehotseq[end]
    shuffle!(onehotseq)
    return (onehotseq, y)
end

function weights(; hidden=256, vocab=50)
    w = Any[]
    push!(w, 0.1*randn(hidden,vocab))#W_hx
    push!(w, 0.1*randn(hidden, hidden))#W_hh
    push!(w, zeros(hidden))#b
    push!(w, 0.1*randn(vocab, hidden))#W_oh
    push!(w, zeros(vocab))#b
    return w
end

function test_example(w; seqlength=5, limit=20)
    seq, ygold = gendata(;seqlength=seqlength, limit=limit)
    println("Sequence: $(map(indmax, seq))")
    
    hT = rnn_sequence(w[1:3], seq)
    ypred = predict(w[4:5], hT)
    
    println("Gold: $(indmax(ygold)), Prediction: $(indmax(ypred))")
    println("")
end

end # module

#=
RNNEXAMPLE.train(;N=2000000, lr=0.001, seqlength=7, limit=50)
(5000,3.9254147954712932,0.05823546687711375)
(15000,3.8347783596082543,0.09440094359889017)
(25000,3.865838169439587,0.09101518162802844)
(35000,3.5961692054202,0.10177853251884963)
(45000,3.591958750481305,0.09625938583918481)
(55000,3.7078481570065813,0.057749964095921785)
(65000,3.662092935174832,0.10045160364338919)
(75000,3.4344444000257703,0.09762542191839016)
(85000,3.5202308574219323,0.1188191211181203)
(95000,3.495033362578429,0.06985608240709526)
(105000,3.545338465844032,0.049726255706462094)
(115000,3.401478302792871,0.08356034017466722)
(125000,3.43937704044176,0.048482881898495034)
(135000,3.373317482719548,0.0783790938216713)
(145000,3.3968358146064377,0.07863264401047583)
(155000,3.3998308365038072,0.058176810199577385)
(165000,3.3854203953236857,0.05996116207383239)
(175000,3.294610861214726,0.05980160736874341)
(185000,3.2955473125602834,0.06772983554593635)
(195000,3.205354931780993,0.044352347305207276)
(205000,3.138209614524962,0.07289154771175269)
(215000,3.136330274070695,0.05219027884713963)
(225000,3.1472323635944233,0.06585019067290764)
(235000,3.2339883176082926,0.037801909117437194)
(245000,3.2237514125329967,0.03909466540726603)
(255000,3.1034637040730155,0.03435449802546742)
(265000,3.01630489214593,0.039125509093287075)
(275000,2.9492079233763255,0.0775554075728231)
(285000,2.924373808650891,0.08119570489621394)
(295000,2.88330660833433,0.045520738204591744)
(305000,2.9801444551347966,0.0720059839435539)
(315000,2.8292788451926425,0.07489431795574271)
(325000,2.9260810182274057,0.06557384998708969)
(335000,2.796193694704773,0.05329545095793585)
(345000,2.8496901913169976,0.07897440943301519)
(355000,2.712887418984886,0.07291655722670402)
(365000,2.765270516987435,0.06446038584847386)
(375000,2.6747343086015882,0.06450701435593854)
(385000,2.593207560038882,0.08195785777298324)
(395000,2.5675051588390225,0.12403903722792178)
(405000,2.440873241602668,0.1286490186185866)
(415000,2.3607913906119866,0.13948185811566524)
(425000,2.422550846616807,0.14491854838751542)
(435000,2.33712184509847,0.1649838142989618)
(445000,2.1728366766728686,0.23305804728313556)
(455000,2.294092413720761,0.21370029615442018)
(465000,2.229902398980125,0.22693723287288248)
(475000,2.0843636254568456,0.29324882274028424)
(485000,2.1690135942845052,0.25494423904147795)
(495000,2.029575497169476,0.2817808574311234)
(505000,2.018764740911919,0.31172281466469576)
(515000,2.0521509456935756,0.32065625642771545)
(525000,1.9296962540268194,0.30291575580380437)
(535000,1.6879912814679325,0.3863069741736311)
(545000,1.8727704996555408,0.4136314713530327)
(555000,1.6648456097910216,0.4396605024667052)
(565000,1.5993401016268955,0.4265090719822417)
(575000,1.5941809610605726,0.4890625982526005)
(585000,1.5426872216335419,0.4989237198379696)
(595000,1.4157123752145753,0.557825588505984)
(605000,1.3032705734715309,0.5745242115672833)
(615000,1.517168941868977,0.5324376419982895)
(625000,1.3540932023513743,0.5474682036807306)
(635000,1.4108951279140025,0.52847086791496)
(645000,1.267215396068737,0.6162706256838634)
(655000,1.1798962320204538,0.6393352147003012)
(665000,1.2345061003259454,0.6481249939547733)
(675000,1.1281916840470916,0.6672409938073572)
(685000,1.0237858905118933,0.7145218218525072)
(695000,0.8914707716711384,0.7717530434428677)
(705000,0.952238328309364,0.72362933798709)
(715000,0.9405367513063994,0.7123276652101643)
(725000,0.8679144953733816,0.7744618942551058)
(735000,0.8550557011342603,0.7791783970643467)
(745000,0.9217059218784082,0.706177293999402)
(755000,0.715502505327051,0.8481938405802976)
(765000,0.8175346060709452,0.7572529026629168)
(775000,0.720734279350491,0.8383031603696639)
(785000,0.7204544240456311,0.7612595785013923)
(795000,0.6605850089156964,0.8389512861661527)
(805000,0.6718430138799281,0.8148635994701189)
(815000,0.6393106793073713,0.8361706527781337)
(825000,0.6666348198783135,0.8435147218150201)
(835000,0.5988272323855302,0.8605988641604053)
(845000,0.6191561860954037,0.8627608081668675)
(855000,0.5832566163231246,0.9016319453333244)
(865000,0.47324692338257224,0.8717556367836733)
(875000,0.45557475839702866,0.9365387594669428)
(885000,0.4668700088971958,0.9122769421372297)
(895000,0.5099601910275182,0.8982515886371417)
(905000,0.47793319039539783,0.927460059393786)
(915000,0.45992243845045605,0.8965995653073635)
(925000,0.4391425468584585,0.8793919722708446)
(935000,0.3897434968359376,0.9662436695667682)
(945000,0.3646875053312192,0.9602772286117665)
(955000,0.3718836517601025,0.9254588034778205)
(965000,0.3392730076665364,0.9510296136308011)
(975000,0.3696439743946451,0.9440675204831033)
(985000,0.37081730190116174,0.9331734713781792)
(995000,0.32695742816382656,0.9343670214841164)
(1005000,0.3299650774822141,0.968463584172169)
(1015000,0.29854385060520233,0.96746614445897)
(1025000,0.3234709585760939,0.9681262164427642)
(1035000,0.2978435021693024,0.9771662701659236)
(1045000,0.2739979235617187,0.9861928981448081)
(1055000,0.26884869311505255,0.9657797242822954)
(1063063,0.24595387097191207,0.9990003244074369)
=#
