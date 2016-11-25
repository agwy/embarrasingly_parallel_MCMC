#Test probit
library(Rcpp)
library(mvtnorm)
library(parallel)

#setwd("R/")

source("probit_funcs.R")
source("MH_MCMC_chain.R")

probit_dimension <- 50
obs_count <- 1e4

simulated_probit_data <- sim_probit(obs_count,probit_dimension)

#Test density functions and use standard GLM functions
glm_test <- glm(simulated_probit_data$obs~simulated_probit_data$design_mat + 0,family = binomial(link=probit))

probit_den(observations = simulated_probit_data$obs, 
           beta = glm_test$coefficients,
           design_mat = simulated_probit_data$design_mat)

augmented_density(observations = simulated_probit_data$obs, 
                  beta = glm_test$coefficients,
                  design_mat = simulated_probit_data$design_mat,
                  to_log=T)


#MCMC approximation
total_iterations <- 1000

source("MH_MCMC_chain.R")
test_MCMC <- MH_MCMC_chain(
  Iterations = total_iterations,
  target_density = augmented_density,
  proposal_sd = 0.01,
  inital_value = as.matrix(rep(0,probit_dimension)),
  observations=simulated_probit_data$obs,
  design_mat=simulated_probit_data$design_mat,
  to_log = T)




#How do our MCMC approximations compare?
plot(rowMeans(test_MCMC)-simulated_probit_data$beta)

augmented_density(observations = simulated_probit_data$obs, 
                  beta = rowMeans(test_MCMC),
                  design_mat = simulated_probit_data$design_mat,
                  to_log=T)




#Parallel implementation

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


#A roughly correct answer with 1000 iterations!
plot(test_nonparametric[,10])
abline(h= simulated_probit_data$beta[10])








