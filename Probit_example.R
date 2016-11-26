#Test probit
library(Rcpp)
library(mvtnorm)
library(parallel)
library(profr)

library(embarrassinglyParallelProbitMCMC, lib.loc="Packages")
# detach?
# source...

probit_dimension <- 50
obs_count <- 1e4

simulated_probit_data <- sim_probit(obs_count,probit_dimension)

## MCMC approximation

total_iterations <- 10000

# R Implementation of a MCMC chain:
source("probit_funcs.R")
source("MH_MCMC_chain.R")
Rprof("timecheck.out")
test_MCMC <- MH_MCMC_chain(
  Iterations = total_iterations,
  target_density = augmented_density,
  proposal_sd = 0.01,
  inital_value = as.matrix(rep(0,probit_dimension)),
  observations=simulated_probit_data$obs,
  design_mat=simulated_probit_data$design_mat,
  to_log = T)
summaryRprof("timecheck.out")

# C implementation of a MCMC chain:
Rprof("timecheck_c.out")
test_MCMC_c <- MCMC_MH(1, 
                       total_iterations, 
                       simulated_probit_data$design_mat, 
                       simulated_probit_data$obs, 
                       rep(0, times=probit_dimension), 
                       0.01)
summaryRprof("timecheck_c.out")
print(test_MCMC_c$Acceptance_rate)


# MCMC results for the true values, R implementation:
plot(rowMeans(test_MCMC)-simulated_probit_data$beta)

augmented_density(observations = simulated_probit_data$obs, 
                  beta = rowMeans(test_MCMC[,-1*(1:1000)]),
                  design_mat = simulated_probit_data$design_mat,
                  to_log=T)

## Parallel implementation

Chain_count <- 8 #Number of subsets

#Break the data into groups
A <- as.list(data.frame(matrix(1:obs_count,ncol=Chain_count)[,1:(Chain_count-1)]))
A[[Chain_count]] <- (tail(A[[Chain_count-1]],1)+1):obs_count


#Run a chain on each group
test3 <- mclapply(A,
                  function(z){t(
                    test_MCMC <- MH_MCMC_chain(
                      Iterations = total_iterations,
                      target_density = augmented_density,
                      proposal_sd = 0.01,
                      inital_value = as.matrix(rep(0,probit_dimension)),
                      observations=simulated_probit_data$obs[z],
                      design_mat=simulated_probit_data$design_mat[z,], ##Pull out those observations 
                      to_log = T,
                      Chain_count = Chain_count)
                    )
                  },
                  mc.cores = min(Chain_count,8)
)


#Implementing the algorithm from the paper to combine chains
source("NonParametric_Density_Product_Estimates.R")
test_nonparametric <- nonparametric_implemetation(test3)


dim(test_nonparametric)

augmented_density(observations = simulated_probit_data$obs, 
                  beta = colMeans(test_nonparametric),
                  design_mat = simulated_probit_data$design_mat,
                  to_log=T)

colMeans(test_nonparametric)

#A roughly correct answer with 1000 iterations!
plot(test_nonparametric[,10])
abline(h= simulated_probit_data$beta[10])

######################################################
## TEST CODE 

#Test density functions and use standard GLM functions

glm_test <- glm(simulated_probit_data$obs~simulated_probit_data$design_mat + 0,family = binomial(link="probit"))


probit_den(observations = simulated_probit_data$obs, 
           beta = glm_test$coefficients,
           design_mat = simulated_probit_data$design_mat)

augmented_density(observations = simulated_probit_data$obs, 
                  beta = glm_test$coefficients,
                  design_mat = simulated_probit_data$design_mat,
                  to_log=T)





