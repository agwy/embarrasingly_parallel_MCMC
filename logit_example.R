#Test logit
library(Rcpp)
library(mvtnorm)
library(parallel)
library(profr)
library(boot)


library(embarrassinglyParallelProbitMCMC, lib.loc="Packages")
#detach("package:embarrassinglyParallelProbitMCMC",unload = TRUE)

#source("embarrassinglyParallelProbitMCMC/R/logit_funcs.R")

logit_dimension <- 50
obs_count <- 4e4
proposal_sd <- 0.03

set.seed(15)
simulated_logit_data <- sim_logit(obs_count,logit_dimension)

## MCMC approximation

total_iterations <- 1e4

# R Implementation of a MCMC chain:
first_time = proc.time()

Rprof(tmp <- tempfile())
test_MCMC <- MH_MCMC_chain(
  Iterations = total_iterations,
  target_density = augmented_density,
  proposal_sd = proposal_sd,
  inital_value = as.matrix(rep(0,logit_dimension)),
  observations=simulated_logit_data$obs,
  design_mat=simulated_logit_data$design_mat,
  to_log = T)
Rprof()
summaryRprof(tmp)
proc.time() - first_time

# C implementation of a MCMC chain:
first_time = proc.time()
test_MCMC_c <- MCMC_MH(1, 
                       total_iterations, 
                       simulated_logit_data$design_mat, 
                       simulated_logit_data$obs, 
                       rep(0, times=logit_dimension), 
                       0.01)
proc.time() - first_time


print(test_MCMC_c$Acceptance_rate)
plot(test_MCMC_c$Result[,1])
abline(h=simulated_logit_data$beta[1])

plot(colMeans(test_MCMC_c$Result),simulated_logit_data$beta)
points(rowMeans(test_MCMC),simulated_logit_data$beta,col="red")


dim(test_MCMC_c$Result)
#glm_test$coefficients

## Parallel implementation

Chain_count <- 8 #Number of subsets

#Break the data into groups
A <- as.list(data.frame(matrix(1:obs_count,ncol=Chain_count)[,1:(Chain_count-1)]))
A[[Chain_count]] <- (tail(A[[Chain_count-1]],1)+1):obs_count


#source("embarrassinglyParallelProbitMCMC/R/MH_MCMC_chain.R")
#Run a chain on each group
first_time = proc.time()
test3 <- mclapply(A,
                  function(z){t(
                    test_MCMC <- MH_MCMC_chain(
                      Iterations = total_iterations,
                      target_density = augmented_density,
                      proposal_sd = proposal_sd,
                      inital_value = as.matrix(rep(0,logit_dimension)),
                      observations=simulated_logit_data$obs[z],
                      design_mat=simulated_logit_data$design_mat[z,], ##Pull out those observations 
                      to_log = T,
                      Chain_count = Chain_count)
                    )
                  },
                  mc.cores = min(Chain_count,8)
)
proc.time() - first_time

#openMP
first_time = proc.time()
test_openMP <- MCMC_MH_parallel(Chain_count, total_iterations, simulated_logit_data$design_mat,
                                simulated_logit_data$obs,rep(0, times=logit_dimension),
                                proposal_sd)
proc.time() - first_time

# Inspect the first beta for the first two chains:
plot(test_openMP$Result[2:total_iterations,1])

# Inspect the second beta:
plot(test_openMP$Result[2:total_iterations,2])

# Inspect the third beta:
plot(test_openMP$Result[2:total_iterations,3])



##############################################################
#Implementing the algorithm from the paper to combine chains
#source("NonParametric_Density_Product_Estimates.R")

#combine the R produced chains
test_nonparametric <- nonparametric_implemetation(test3)
dim(test_nonparametric)

colMeans(test_nonparametric)

#A roughly correct answer with 10000 iterations!
plot(test_nonparametric[,10])
abline(h= simulated_logit_data$beta[10], col="red")

# #combine the C produced chains
# test_OpenMP_list = list()
# total_iterations1 = total_iterations + 1
# for(i in 1:Chain_count){
#   test_OpenMP_list[[i]] = as.matrix(test_openMP$Result[((i-1)*total_iterations1+1):(i*total_iterations1),])
#   test_OpenMP_list[[i]] = test_OpenMP_list[[i]][-total_iterations1,]
# }
# 
# 
# test_nonparametric_c <- nonparametric_implemetation((test_OpenMP_list))
# 
# plot(test_nonparametric_c[,10])
# abline(h= simulated_logit_data$beta[10], col="red")


############Compare 'full' posterior with the 'combined' posterior 
#means for the single chain run on all data
colMeans(t(test_MCMC))

#means for the combined chain 
colMeans(test_nonparametric)

#true beta used in the simulation
simulated_logit_data$beta

######################################################
## TEST CODE 

#Test density functions and use standard GLM functions

# glm_test <- glm(simulated_logit_data$obs~simulated_logit_data$design_mat + 0,family = binomial(link="logit"))
# 
# 
# logit_den(observations = simulated_logit_data$obs, 
#            beta = glm_test$coefficients,
#            design_mat = simulated_logit_data$design_mat)
# 
# augmented_density(observations = simulated_logit_data$obs, 
#                   beta = glm_test$coefficients,
#                   design_mat = simulated_logit_data$design_mat,
#                   to_log=T)





