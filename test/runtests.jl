#Uncomment this for better testing
using Random; Random.seed!(1)
@time include("base.jl")
#TODO include("broadcast.jl")
@time include("cat.jl")
@time include("core.jl")
##@time include("getindex.jl")
@time include("iterate.jl")
@time include("linearalgebra.jl")
#TODO include("macros.jl")
@time include("math.jl")
#TODO include("params.jl")
@time include("specialfunctions.jl")
@time include("statistics.jl")
@time include("rosenbrock.jl")
@time include("highorder.jl")
@time include("neuralnet.jl")

# MASTER:                                                                  COREHACK:
# julia> include("runtests.jl")                                            julia> include("runtests.jl")					     
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# base          |  111    111						 base          |  111    111					     
#  53.581755 seconds (116.08 M allocations: 5.798 GiB, 4.99% gc time)	  41.620695 seconds (101.53 M allocations: 5.084 GiB, 5.66% gc time)  
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# cat           |   20     20						 cat           |   20     20					     
#  10.165758 seconds (18.04 M allocations: 946.009 MiB, 3.28% gc time)	   9.109428 seconds (15.91 M allocations: 836.354 MiB, 4.80% gc time) 
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# core          |    6      6						 core          |    6      6					     
#   5.268114 seconds (12.50 M allocations: 643.580 MiB, 5.12% gc time)	   4.759409 seconds (10.22 M allocations: 532.706 MiB, 6.44% gc time) 
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# iterate       |   14     14						 iterate       |   14     14					     
#   8.412887 seconds (15.30 M allocations: 799.478 MiB, 5.06% gc time)	   6.158372 seconds (11.90 M allocations: 627.248 MiB, 6.28% gc time) 
# Test Summary: | Pass  Broken  Total					 Test Summary: | Pass  Broken  Total				     
# LinearAlgebra |   34      14     48					 LinearAlgebra |   34      14     48				     
#  15.546335 seconds (33.19 M allocations: 1.673 GiB, 6.08% gc time)	  14.266001 seconds (28.39 M allocations: 1.440 GiB, 5.19% gc time)   
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# math          |   62     62						 math          |   62     62					     
#  59.146167 seconds (132.70 M allocations: 6.698 GiB, 7.76% gc time)	  51.213833 seconds (100.58 M allocations: 5.092 GiB, 7.47% gc time)  
# Test Summary:    | Pass  Total						 Test Summary:    | Pass  Total					     
# specialfunctions |   20     20						 specialfunctions |   20     20					     
#  19.632797 seconds (41.75 M allocations: 2.117 GiB, 9.71% gc time)	  15.259193 seconds (30.56 M allocations: 1.554 GiB, 7.91% gc time)   
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# statistics    |   22     22						 statistics    |   22     22					     
#  12.828401 seconds (23.18 M allocations: 1.188 GiB, 7.94% gc time)	  11.161135 seconds (20.08 M allocations: 1.035 GiB, 7.48% gc time)   
#   2.158933 seconds (2.65 M allocations: 88.297 MiB, 6.35% gc time)	   5.932944 seconds (3.93 M allocations: 165.509 MiB, 4.93% gc time)  
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# rosenbrock    |    8      8						 rosenbrock    |    8      8					     
#   8.924719 seconds (13.78 M allocations: 667.723 MiB, 7.28% gc time)	  12.462551 seconds (11.82 M allocations: 582.442 MiB, 7.33% gc time) 
# Test Summary: | Pass  Total						 Test Summary: | Pass  Broken  Total				     
# highorder     |   15     15						 highorder     |   14       1     15				     
#   2.841089 seconds (3.44 M allocations: 187.509 MiB, 10.01% gc time)	   2.911816 seconds (3.85 M allocations: 194.752 MiB, 6.46% gc time)  
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# neuralnet     |    6      6						 neuralnet     |    6      6					     
#   7.923417 seconds (12.10 M allocations: 631.913 MiB, 7.32% gc time)	   5.997711 seconds (9.06 M allocations: 478.741 MiB, 7.12% gc time)  
									 								     
# julia> include("runtests.jl")						 julia> include("runtests.jl")					     
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# base          |  111    111						 base          |  111    111					     
#   4.892052 seconds (4.20 M allocations: 238.570 MiB, 3.93% gc time)	   5.221830 seconds (5.60 M allocations: 305.606 MiB, 4.62% gc time)  
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# cat           |   20     20						 cat           |   20     20					     
#   3.009139 seconds (3.20 M allocations: 183.307 MiB, 4.58% gc time)	   3.058421 seconds (3.63 M allocations: 203.215 MiB, 6.09% gc time)  
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# core          |    6      6						 core          |    6      6					     
#   0.476055 seconds (429.55 k allocations: 22.139 MiB, 3.76% gc time)	   0.509822 seconds (497.30 k allocations: 25.423 MiB, 3.72% gc time) 
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# iterate       |   14     14						 iterate       |   14     14					     
#   3.119049 seconds (3.73 M allocations: 204.028 MiB, 5.18% gc time)	   3.154161 seconds (4.10 M allocations: 220.886 MiB, 6.45% gc time)  
# Test Summary: | Pass  Broken  Total					 Test Summary: | Pass  Broken  Total				     
# LinearAlgebra |   34      14     48					 LinearAlgebra |   34      14     48				     
#   1.566442 seconds (1.49 M allocations: 84.065 MiB, 4.50% gc time)	   1.661380 seconds (1.88 M allocations: 102.103 MiB, 5.49% gc time)  
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# math          |   62     62						 math          |   62     62					     
#   5.662640 seconds (10.33 M allocations: 520.636 MiB, 8.93% gc time)	   5.397801 seconds (10.30 M allocations: 521.037 MiB, 8.54% gc time) 
# Test Summary:    | Pass  Total						 Test Summary:    | Pass  Total					     
# specialfunctions |   20     20						 specialfunctions |   20     20					     
#   2.577594 seconds (5.12 M allocations: 258.258 MiB, 9.04% gc time)	   2.655762 seconds (5.09 M allocations: 257.343 MiB, 15.07% gc time) 
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# statistics    |   22     22						 statistics    |   22     22					     
#   0.219993 seconds (20.01 k allocations: 1.062 MiB)			   0.260460 seconds (20.79 k allocations: 1.094 MiB)		     
#   2.136295 seconds (2.65 M allocations: 88.297 MiB, 4.83% gc time)	   5.718395 seconds (3.93 M allocations: 165.509 MiB, 3.26% gc time)  
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# rosenbrock    |    8      8						 rosenbrock    |    8      8					     
#   4.003694 seconds (4.73 M allocations: 196.375 MiB, 4.96% gc time)	   7.609773 seconds (6.32 M allocations: 287.820 MiB, 3.84% gc time)  
# Test Summary: | Pass  Total						 Test Summary: | Pass  Broken  Total				     
# highorder     |   15     15						 highorder     |   14       1     15				     
#   0.320512 seconds (161.29 k allocations: 7.923 MiB)			   0.523189 seconds (576.37 k allocations: 28.446 MiB, 5.60% gc time) 
# Test Summary: | Pass  Total						 Test Summary: | Pass  Total					     
# neuralnet     |    6      6						 neuralnet     |    6      6					     
#   0.292202 seconds (165.82 k allocations: 8.772 MiB, 18.74% gc time)	   0.297534 seconds (177.61 k allocations: 9.496 MiB)                 

