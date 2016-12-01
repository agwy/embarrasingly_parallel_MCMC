
## Driver file to perform a single random walk MCMC run with spherical symmetrical proposal matrix for 
## the posterior of the parameters of a logisitic regression model. The computation of the 
## log-likelihood is parallelized on GPUs, i.e. matrix multiplications and computation of 
## the logistic function.

library(mvtnorm)
library(profr)
#detach("package:OxWaSP",unload = TRUE) # TODO: modify

# Load the package where the GPU MCMC library is installed:
library(GPULogitMCMC, lib.loc = "Packages/")

is.loaded("GPU_Probit_LL") # Check if the package is loaded


# Specifies the number of observations and the number of parameters in the logit model:
num_obs = 2e4      # The number of observations
num_dim = 50       # The number of parameters
num_its = 2e4      # The number of iterations.
proposal_sd = .03  # The proposal standard deviation for the random walk MCMC algorithm.

set.seed(15)
test_data <- sim_logit(num_obs, num_dim)
#test_data <- sim_probit(num_obs, num_dim)

# Check if the loglikelihood is computed correctly:
LL_logit_GPU(Observations = test_data$obs, Beta = test_data$beta, Data_Matrix = t(test_data$design_mat))
logit_den(observations = test_data$obs,
          beta = test_data$beta,
          design_mat = test_data$design_mat,to_log = T)



### Testing full semi-MCMC chain on GPU
# Note that the implementation requires the data_matrix to be transposed:
system.time(test_GPU_MCMC <-GPU_logit_MCMC_C(
  Observations = test_data$obs,
  Beta = rep(0, num_dim),
  Data_Matrix = t(test_data$design_mat),
  Iterations = num_its,
  Proposal_sd = proposal_sd))
# For 1e4 obs, 50 dim, 1e4 its, prop_sd = .03:
#   user  system elapsed 
# 2.149   0.216   2.364 
# For 2e4 obs, 50 dim, 2e4 its, prop_sd = .03:
# user  system elapsed 
# 7.591   1.096   8.684 
# (previous run)  user  system elapsed 
# 3.891   0.473   4.361 
# Note however that this is with a uniform prior.

# Check output and mean of the beta's after a a burn-in:
test_GPU_MCMC$Acceptance_rate
test_GPU_MCMC$Chain_Output[, num_its]
rowMeans(test_GPU_MCMC$Chain_Output[,floor(num_its/6):num_its])

# An R implementation:
system.time(test_standard <- MH_MCMC_chain(Iterations = num_its,
                                           target_density = logit_den,
                                           proposal_sd = proposal_sd,
                                           inital_value = as.matrix(rep(0,num_dim)),
                                           observations = test_data$obs,
                                           design_mat = test_data$design_mat,to_log = T))
# Here
# For 1e4 obs, 50 dim, 1e4 its, prop_sd = .03:
# user  system elapsed 
# 60.921   0.053  60.977 
# For 2e4 obs, 50 dim, 2e4 its, prop_sd = .03:
# user  system elapsed 
# 118.196   0.128 118.273 

# Do they yield similar results?
rowMeans(test_GPU_MCMC$Chain_Output)
rowMeans(test_standard)
plot(rowMeans(test_GPU_MCMC$Chain_Output), rowMeans(test_standard))

# Profiling the standard MCMC chain to have some idea about the bottlenecks:
Rprof("timecheck.out")
test_standard <- MH_MCMC_chain(Iterations = num_its,
                               target_density = logit_den,
                               proposal_sd = proposal_sd,
                               inital_value = as.matrix(rep(0,num_dim)),
                               observations = test_data$obs,
                               design_mat = test_data$design_mat,to_log = T
)
summaryRprof("timecheck.out")
# For 2e4 obs, 50 dim, 2e4 its, prop_sd = .03:
# $by.self
# self.time self.pct total.time total.pct
# "%*%"                72.08    60.94      72.08     60.94
# "plogis"             24.96    21.10      97.04     82.04
# "target_density"     18.40    15.56     117.38     99.24
# "!"                   1.22     1.03       1.22      1.03
# "sum"                 0.52     0.44       0.52      0.44
# "MH_MCMC_chain"       0.42     0.36     118.26     99.98
# "rnorm"               0.20     0.17       0.20      0.17
# "runif"               0.20     0.17       0.20      0.17
# "-"                   0.18     0.15       0.18      0.15
# "+"                   0.04     0.03       0.04      0.03
# ">"                   0.04     0.03       0.04      0.03
# "exp"                 0.02     0.02       0.02      0.02
# 
# $by.total
# total.time total.pct self.time self.pct
# "MH_MCMC_chain"      118.26     99.98      0.42     0.36
# "target_density"     117.38     99.24     18.40    15.56
# "plogis"              97.04     82.04     24.96    21.10
# "%*%"                 72.08     60.94     72.08    60.94
# "!"                    1.22      1.03      1.22     1.03
# "sum"                  0.52      0.44      0.52     0.44
# "rnorm"                0.20      0.17      0.20     0.17
# "runif"                0.20      0.17      0.20     0.17
# "-"                    0.18      0.15      0.18     0.15
# "+"                    0.04      0.03      0.04     0.03
# ">"                    0.04      0.03      0.04     0.03
# "exp"                  0.02      0.02      0.02     0.02
# "MH_MCMC_chain         0.02      0.02      0.00     0.00

### Comparison with a cuBLAS implementation:
# Note, don't transpose the data_matrix:
system.time(test_cuBLAS <- GPU_logit_MCMC_cuBLAS( Observations = test_data$obs, 
                                      Beta = rep(0, num_dim),
                                      Data_Matrix = test_data$design_mat,
                                      Iterations = num_its,
                                      Proposal_sd = proposal_sd))

rowMeans(test_cuBLAS$Chain_Output)
plot(rowMeans(test_GPU_MCMC$Chain_Output), rowMeans(test_cuBLAS$Chain_Output))

#######################################################################################
### Time comparisons that try to average out overheads:
num_obs = 2e4      # The number of observations
num_dim = 50       # The number of parameters
proposal_sd = .03  # The proposal standard deviation for the random walk MCMC algorithm.
base_num_its = 1e4
num_its_R = seq_len(5)
num_its_GPU = seq_len(8)
times = list(times_R = rep(0, times = length(num_its_R)), 
             times_GPU_1 = rep(0, length(num_its_GPU)), 
             times_GPU_cuBLAS = rep(0, length(num_its_GPU)))

set.seed(15)
test_data <- sim_logit(num_obs, num_dim)

# First GPU implementation:
set.seed(16)
for(i in num_its_GPU){
  time = system.time(GPU_logit_MCMC_C(
    Observations = test_data$obs,
    Beta = rep(0, num_dim),
    Data_Matrix = t(test_data$design_mat),
    Iterations = i * base_num_its,
    Proposal_sd = proposal_sd))
  print(paste("iteration ", i, ":"))
  print(time)
  times$times_GPU_1[i] = time[3]
}
# Per step of 10000 its:
# 4.944 10.130 15.285 19.523 24.827 29.022 33.516 37.596

# cuBLAS implementation
set.seed(16)
for(i in num_its_GPU){
  time = system.time(GPU_logit_MCMC_cuBLAS(
    Observations = test_data$obs,
    Beta = rep(0, num_dim),
    Data_Matrix = test_data$design_mat,
    Iterations = i * base_num_its,
    Proposal_sd = proposal_sd))
  print(paste("iteration ", i, ":"))
  print(time)
  times$times_GPU_cuBLAS[i] = time[3]
}
# Per step of 10000 its:
# 1.738  3.817  5.948  8.341  9.760 12.106 13.819 15.567

# R implementation:
set.seed(16)
for(i in num_its_R){
  time = system.time(MH_MCMC_chain(Iterations = i * base_num_its,
                                  target_density = logit_den,
                                  proposal_sd = proposal_sd,
                                  inital_value = as.matrix(rep(0,num_dim)),
                                  observations = test_data$obs,
                                  design_mat = test_data$design_mat,to_log = T))
  print(paste("iteration ", i, ":"))
  print(time)
  times$times_R[i] = time[3]
}
# Per step of 10000 its on Petya's local computer:
# 26.191  52.156  77.064 105.916 127.943
