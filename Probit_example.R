#Test probit
library(Rcpp)
library(mvtnorm)
library(parallel)
library(profr)
library(boot)


library(embarrassinglyParallelProbitMCMC, lib.loc="Packages")
detach("package:embarrassinglyParallelProbitMCMC",unload = TRUE)
# detach?
# source...
source("embarrassinglyParallelProbitMCMC/R/probit_funcs.R")

probit_dimension <- 50
obs_count <- 8e4
proposal_sd <- 0.03

set.seed(15)
simulated_probit_data <- sim_probit(obs_count,probit_dimension)

## MCMC approximation

total_iterations <- 1e3

# R Implementation of a MCMC chain:
first_time = proc.time()

Rprof(tmp <- tempfile())
test_MCMC <- MH_MCMC_chain(
  Iterations = total_iterations,
  target_density = augmented_density,
  proposal_sd = proposal_sd,
  inital_value = as.matrix(rep(0,probit_dimension)),
  observations=simulated_probit_data$obs,
  design_mat=simulated_probit_data$design_mat,
  to_log = T)
Rprof()
summaryRprof(tmp)
proc.time() - first_time

# C implementation of a MCMC chain:
first_time = proc.time()
Rprof("timecheck.out")
test_MCMC_c <- MCMC_MH(1, 
                       total_iterations, 
                       simulated_probit_data$design_mat, 
                       simulated_probit_data$obs, 
                       rep(0, times=probit_dimension), 
                       0.01)
summaryRprof("timecheck.out")

proc.time() - first_time


print(test_MCMC_c$Acceptance_rate)
plot(test_MCMC_c$Result[,1])


plot(colMeans(test_MCMC_c$Result),simulated_probit_data$beta)
points(rowMeans(test_MCMC),simulated_probit_data$beta,col="red")


plot(test_MCMC_c$Result[,1])
abline(h=simulated_probit_data$beta[1])

dim(test_MCMC_c$Result)
glm_test$coefficients

## Parallel implementation

Chain_count <- 8 #Number of subsets

#Break the data into groups
A <- as.list(data.frame(matrix(1:obs_count,ncol=Chain_count)[,1:(Chain_count-1)]))
A[[Chain_count]] <- (tail(A[[Chain_count-1]],1)+1):obs_count


source("embarrassinglyParallelProbitMCMC/R/MH_MCMC_chain.R")
#Run a chain on each group
first_time = proc.time()
Rprof("timecheck_parallel.out")
test3 <- mclapply(A,
                  function(z){t(
                    test_MCMC <- MH_MCMC_chain(
                      Iterations = total_iterations,
                      target_density = augmented_density,
                      proposal_sd = proposal_sd,
                      inital_value = as.matrix(rep(0,probit_dimension)),
                      observations=simulated_probit_data$obs[z],
                      design_mat=simulated_probit_data$design_mat[z,], ##Pull out those observations 
                      to_log = T,
                      Chain_count = Chain_count)
                    )
                  },
                  mc.cores = min(Chain_count,8)
)
summaryRprof("timecheck_parallel.out")
proc.time() - first_time

#openMP
first_time = proc.time()
Rprof("timecheck_parallel_c.out")
test_openMP <- MCMC_MH_parallel(Chain_count, total_iterations, simulated_probit_data$design_mat,
                                simulated_probit_data$obs,rep(0, times=probit_dimension),
                                proposal_sd)
summaryRprof("timecheck_parallel_c.out")
print("Time measured with time.proc:")
proc.time() - first_time
back_up <- test_openMP

# Inspect the first beta for the first two chains:
plot(test_openMP$Result[2:total_iterations,1])
plot(1:10000, test_openMP$Result[10002:20001,1])
# Inspect the second beta:
plot(test_openMP$Result[2:total_iterations,2])
plot(1:10000, test_openMP$Result[10002:20001,2])
# Inspect the third beta:
plot(test_openMP$Result[2:total_iterations,3])
plot(1:10000, test_openMP$Result[10002:20001,3])

#Implementing the algorithm from the paper to combine chains
source("NonParametric_Density_Product_Estimates.R")
test_nonparametric <- nonparametric_implemetation(test3)


dim(test_nonparametric)

colMeans(test_nonparametric)

#A roughly correct answer with 1000 iterations!
plot(test_nonparametric[,10])
abline(h= simulated_probit_data$beta[10])

######################################################
## TEST CODE 

#Test density functions and use standard GLM functions

glm_test <- glm(simulated_probit_data$obs~simulated_probit_data$design_mat + 0,family = binomial(link="logit"))


probit_den(observations = simulated_probit_data$obs, 
           beta = glm_test$coefficients,
           design_mat = simulated_probit_data$design_mat)

augmented_density(observations = simulated_probit_data$obs, 
                  beta = glm_test$coefficients,
                  design_mat = simulated_probit_data$design_mat,
                  to_log=T)





