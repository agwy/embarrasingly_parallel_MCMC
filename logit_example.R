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

total_iterations <- 2e4

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
                       proposal_sd)
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

#combine the C produced chains
burnin=0.2*total_iterations #introduce some burnin when dividing the openMP matrix
test_OpenMP_list = list()
total_iterations1 = total_iterations + 1
for(i in 1:Chain_count){
  test_OpenMP_list[[i]] = as.matrix(test_openMP$Result[((i-1)*total_iterations1+1+burnin):(i*total_iterations1),])
  test_OpenMP_list[[i]] = test_OpenMP_list[[i]][-total_iterations1,]
}


test_nonparametric_c <- nonparametric_implemetation((test_OpenMP_list))
# 
# plot(test_nonparametric_c[,10])
# abline(h= simulated_logit_data$beta[10], col="red")


############Compare 'full' posterior with the 'combined' posterior 
#means for the single chain run on all data
colMeans(t(test_MCMC))
apply(t(test_MCMC), 2, sd)

#means for the combined chain 
colMeans(test_nonparametric)
apply(test_nonparametric, 2, sd)

#true beta used in the simulation
simulated_logit_data$beta

plot(test_MCMC_c$Result[,1], test_MCMC_c$Result[,2], ylim=c(0,5), xlim=c(-.5,.5))
points(test_openMP$Result[,1], test_openMP$Result[,2], col="red",ylim=c(0,5), xlim=c(-.5,.5))
points(test_nonparametric[,1], test_nonparametric[,2], col="blue",ylim=c(0,5), xlim=c(-.5,.5))


####################################################
#Combine the chains using the parametric algorithm

test_parametric = parametric_implementation(test3, burnin=0.2*total_iterations)
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

########################################################
#some plots

library(MASS)
z = kde2d(test_MCMC_c$Result[(0.5*total_iterations):total_iterations,1], 
          test_MCMC_c$Result[(0.5*total_iterations):total_iterations,2], n=50)#burnin 50%
z2 = kde2d(test_openMP$Result[(0.2*total_iterations):total_iterations,1], 
           test_openMP$Result[(0.2*total_iterations):total_iterations,2], n=50)#burnin 20%

z3 = kde2d(test_nonparametric_c[,1], test_nonparametric_c[,2], n=50)#no burnin now!!!
z_par = kde2d(test_parametric[,1], test_parametric[,2], n=50)#burnin incorporated in function, 20%
contour(z, xlim=c(-0.5,1.5), ylim=c(1,3), col="black")
par(new=T)
contour(z2, xlim=c(-0.5,1.5), ylim=c(1,3), col="red")
par(new=T)
contour(z3,  xlim=c(-0.5,1.5), ylim=c(1,3), col="blue")
par(new=T)
contour(z_par, xlim=c(-0.5,1.5), ylim=c(1,3), col="yellow")


z4 = kde2d(test_openMP$Result[(total_iterations+2):(2*total_iterations+2),1], 
           test_openMP$Result[(total_iterations+2):(2*total_iterations+2),2], n=50)
z5 = kde2d(test_openMP$Result[(2*total_iterations+3):(3*total_iterations+3),1], 
           test_openMP$Result[(2*total_iterations+3):(3*total_iterations+3),2], n=50)
contour(z, xlim=c(0,1), ylim=c(1,3), col="black")
par(new=T)
contour(z2, xlim=c(0,1), ylim=c(1,3), col="red")
par(new=T)
contour(z4, xlim=c(0,1), ylim=c(1,3), col="green")
par(new=T)
contour(z5, xlim=c(0,1), ylim=c(1,3), col="purple")

####does this make any sense?
####data ellipses
library(car)
dataEllipse(test_MCMC_c$Result[(0.5*total_iterations):total_iterations,1], 
              test_MCMC_c$Result[(0.5*total_iterations):total_iterations,2], levels=0.9,
            col="black", xlim=c(-0.5,1), ylim=c(1,3))
par(new=T)
dataEllipse(test_openMP$Result[(0.2*total_iterations):total_iterations,1], 
            test_openMP$Result[(0.2*total_iterations):total_iterations,2], levels=0.9,
            plot.points = FALSE, col="red", xlim=c(-0.5,1), ylim=c(1,3))
par(new=T)
dataEllipse(test_nonparametric_c[,1], test_nonparametric_c[,2], levels=0.9,
            plot.points = FALSE, col="blue", xlim=c(-0.5,1), ylim=c(1,3))
par(new=T)
dataEllipse(test_parametric[,1], test_parametric[,2], levels=0.9,
            plot.points = FALSE, col="yellow", xlim=c(-0.5,1), ylim=c(1,3))
