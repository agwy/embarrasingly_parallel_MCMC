# R CMD build embarrassinglyParallelProbitMCMC/
# R CMD INSTALL -l Packages/ embarrassinglyParallelProbitMCMC_0.1.0.tar.gz
#Logistic regression is used to demonstrate the speed of the parallelization#

#at first Probit regression was implemented so there could be some
#commented lines of code related to Probit;

library(Rcpp)
library(mvtnorm)
library(parallel)
library(profr)
library(boot)

library(embarrassinglyParallelProbitMCMC, lib.loc="Packages")
#detach("package:embarrassinglyParallelProbitMCMC",unload = TRUE)
#########################################################################
#### simulate the data #####

logit_dimension <- 50 #number of parameters
obs_count <- 4e4 #number of observations

set.seed(15)
simulated_logit_data <- sim_logit(obs_count,logit_dimension)

#########################################################################
#### MCMC approximation ####

total_iterations <- 2e4 #number of iterations
proposal_sd <- 0.01 #proposal standard deviation for the random walk MCMC algorithm.

#For proposal standard deviation 0.03 - the acc rate is only 1% for the single chain;
#So, proposal standard deviation 0.01 is used instead and the acc rate for the single chain is about 12%;
#Initial value is taken to be a vector of 0s, i.e. the markov chain starts from 0 for each parameter;

# R Implementation of a single MCMC chain - full data:
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
##########################################################
# > summaryRprof(tmp)
# $by.self
# self.time self.pct total.time total.pct
# "%*%"                55.96    53.38      55.96     53.38
# "logit_den"          26.50    25.28     104.10     99.29
# "plogis"             20.06    19.13      76.02     72.51
# "!"                   0.76     0.72       0.76      0.72
# "sum"                 0.66     0.63       0.66      0.63
# "MH_MCMC_chain"       0.32     0.31     104.84    100.00
# "-"                   0.18     0.17       0.18      0.17
# "runif"               0.12     0.11       0.12      0.11
# "dnorm"               0.10     0.10       0.12      0.11
# "target_density"      0.08     0.08     104.32     99.50
# "rnorm"               0.08     0.08       0.08      0.08
# "list"                0.02     0.02       0.02      0.02
# 
# $by.total
# total.time total.pct self.time self.pct
# "MH_MCMC_chain"      104.84    100.00      0.32     0.31
# "target_density"     104.32     99.50      0.08     0.08
# "logit_den"          104.10     99.29     26.50    25.28
# "plogis"              76.02     72.51     20.06    19.13
# "%*%"                 55.96     53.38     55.96    53.38
# "!"                    0.76      0.72      0.76     0.72
# "sum"                  0.66      0.63      0.66     0.63
# "-"                    0.18      0.17      0.18     0.17
# "runif"                0.12      0.11      0.12     0.11
# "dnorm"                0.12      0.11      0.10     0.10
# "rnorm"                0.08      0.08      0.08     0.08
# "list"                 0.02      0.02      0.02     0.02
# 
# $sample.interval
# [1] 0.02
# 
# $sampling.time
# [1] 104.84
# 
# > proc.time() - first_time
# user  system elapsed 
# 104.889   0.015 105.191 
##########################################################

# C implementation of a single MCMC chain - full data:
first_time = proc.time()
test_MCMC_c <- MCMC_MH(1,
                       total_iterations,
                       simulated_logit_data$design_mat,
                       simulated_logit_data$obs,
                       rep(0, times=logit_dimension),
                       proposal_sd)
proc.time() - first_time
# > proc.time() - first_time
# user  system elapsed
# 65.976   0.011  66.111

#the C implementation is almost twice faster than the R (single chain; full data);

print(test_MCMC_c$Acceptance_rate) #12%

#visually check convergence; no burnin on plot;
plot(test_MCMC_c$Result[,1])
abline(h=simulated_logit_data$beta[1])

#plot the estimated posterior mean for C (in black) and R (in red) chains
plot(colMeans(test_MCMC_c$Result),simulated_logit_data$beta)
points(rowMeans(test_MCMC),simulated_logit_data$beta,col="red")
abline(a=0, b = 1)

#we can also compare our estimated mean to the R glm estimated regression coefficients
glm_test <- glm(simulated_logit_data$obs~simulated_logit_data$design_mat + 0,family = binomial(link="logit"))
plot(glm_test$coefficients,colMeans(test_MCMC_c$Result))
abline(a=0,b=1)


##################################################################################
## Parallel implementation

Chain_count <- 8 #Number of subsets

############################################################
####Parallel implementation in R using the function mclapply
############################################################

#Break the data into groups
A <- as.list(data.frame(matrix(1:obs_count,ncol=Chain_count)[,1:(Chain_count-1)]))
A[[Chain_count]] <- (tail(A[[Chain_count-1]],1)+1):obs_count

#Run a chain on each group, containing (1/M) of the data
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
# > proc.time() - first_time
# user  system elapsed
# 218.054   1.308  33.294

#just out of interest, we ran the code for 16 chains
#and the time is
# > proc.time() - first_time
# user  system elapsed
# 292.792  49.299  70.833
# > first_time = proc.time()

############################################################
####Parallel implementation using openMP
############################################################

#this was run on our machines and the number of threads is fixed to 8,
#on our machines the maximum number of threads is 8:
first_time = proc.time()
test_openMP <- MCMC_MH_parallel(Chain_count, total_iterations, simulated_logit_data$design_mat,
                                simulated_logit_data$obs,rep(0, times=logit_dimension),
                                proposal_sd)
proc.time() - first_time
#is.loaded("openMP")
#dyn.load("Packages/embarrassinglyParallelProbitMCMC/libs/embarrassinglyParallelProbitMCMC.so")
#source("embarrassinglyParallelProbitMCMC/R/C_Wrapper_MCMC.R")

# > proc.time() - first_time
# user  system elapsed
# 230.551   0.083  31.553

#we wanted to fix the number of threads to 4 and check the time again
#this line was added to our openMP C function
#omp_set_num_threads(4);
#and the run time is
# > proc.time() - first_time
# user  system elapsed 
# 195.658   0.044  49.423 
#which confirms we are using multithreading and the mclapply function does as well;  

#again out of interest, we ran the code for 16 chains
#and the time is
# > proc.time() - first_time
# user  system elapsed
# 195.821   0.093  29.514

# Inspect the first beta for the first chain to visually check convergence
# We expect some deviations since only (1/M)th of the data is used:
plot(test_openMP$Result[2:total_iterations,1])

# Inspect the second beta:
plot(test_openMP$Result[2:total_iterations,2])

# Inspect the third beta:
plot(test_openMP$Result[2:total_iterations,3])


####################################################################
#Combine the chains using the Nonparametric algorithm from the paper
####################################################################

##### R produced chains #####
#combine the R produced chains; 20% burnin taken
first_time = proc.time()
test_nonparametric <- nonparametric_implementation(test3, burnin=0.2*total_iterations)
proc.time()-first_time
# > proc.time()-first_time
# user  system elapsed
# 39.307   0.076  39.494

dim(test_nonparametric)

#does the combined chain converge to the truth?
plot(colMeans(test_nonparametric), simulated_logit_data$beta)
abline(a=0, b=1)

#Inspect the first beta for the combibed chain
plot(test_nonparametric[,1])
abline(h= simulated_logit_data$beta[1], col="red")

##### C produced chains #####
#combine the C produced chains

#first make a list of matrices from the openMP output which is a matrix
test_OpenMP_list = list()
total_iterations1 = total_iterations + 1
for(i in 1:Chain_count){
  test_OpenMP_list[[i]] = as.matrix(test_openMP$Result[((i-1)*total_iterations1+1):(i*total_iterations1),])
  test_OpenMP_list[[i]] = test_OpenMP_list[[i]][-total_iterations1,]
}

#combine the R produced chains; 20% burnin taken
test_nonparametric_c <- nonparametric_implementation(test_OpenMP_list, burnin=0.2*total_iterations)

#Inspect the first beta for the combibed chain
plot(test_nonparametric_c[,10])
abline(h= simulated_logit_data$beta[10], col="red")


####Compare 'full' posterior with the 'combined' posterior ####

#mean and sd for the single chain, full data
colMeans(t(test_MCMC))
apply(t(test_MCMC), 2, sd)

#means for the combined chain using the Nonparametric algorithm
colMeans(test_nonparametric)
apply(test_nonparametric, 2, sd)

#true beta used in the simulation
simulated_logit_data$beta

####################################################################
#Combine the chains using the Parametric algorithm from the paper
####################################################################

# using the R produced chains
first_time=proc.time()
test_parametric = parametric_implementation(test3, burnin=0.2*total_iterations)
proc.time() - first_time
# > proc.time() - first_time
# user  system elapsed
# 11.940   0.001  11.976

#mean and sd for the combined chain using the Parametric algorithm
colMeans(test_parametric)
apply(test_parametric, 2, sd)

####################################################################
#Combine the chains using the SemiParametric algorithm from the paper
#it takes a long time to run since it combines the parametric and nonparametric algorithms
####################################################################

# using the R produced chains
first_time=proc.time()
test_semiparametric = Semiparametric_implementation(test3, burnin=0.2*total_iterations)
proc.time() - first_time
# > proc.time() - first_time
# user  system elapsed
# 231.415   0.464 232.598


#mean and sd for the combined chain using the Parametric algorithm
colMeans(test_semiparametric)
apply(test_semiparametric, 2, sd)

###################################################################
#some plots to illustrate the chain's behaviour

library(MASS)

#calculate the joint density for beta1 and beta2 chains for:
#(1) the full chain
full_c = kde2d(test_MCMC_c$Result[(0.5*total_iterations):total_iterations,1],
          test_MCMC_c$Result[(0.5*total_iterations):total_iterations,2], n=30)#burnin 50%

#(2) the chains on subsets
openMP_kde = list()
for(i in 1:Chain_count){
  openMP_kde[[i]] = kde2d(test_OpenMP_list[[i]][(0.2*total_iterations):total_iterations,1],
                     test_OpenMP_list[[i]][(0.2*total_iterations):total_iterations,2], n=30)
}

#(3) the combined chain using the nonparametric algorithm
comb_nonpar = kde2d(test_nonparametric[,1], test_nonparametric[,2], n=30)

#(4) the combined chain using the parametric algorithm
comb_par = kde2d(test_parametric[,1], test_parametric[,2], n=30)

#(5) the combined chain using the semiparametric algorithm
comb_semipar = kde2d(test_semiparametric[,1], test_semiparametric[,2], n=30)

#plot in a contour plot the full chain and the combined chains
contour(full_c, xlim=c(-0.5,1.5), ylim=c(1,3), col="black")
par(new=T)
contour(comb_nonpar,  xlim=c(-0.5,1.5), ylim=c(1,3), col="blue")
par(new=T)
contour(comb_par, xlim=c(-0.5,1.5), ylim=c(1,3), col="yellow")
#par(new=T)
#contour(comb_semipar, xlim=c(-0.5,1.5), ylim=c(1,3), col="pink")
#the semiparametric method gives similar results to the parametric (for this example)


#the Nonparametric chain and the openMP subsets (all M); density for beta1 and beta2
contour(comb_nonpar,  xlim=c(-0.5,1.5), ylim=c(1,3), col="blue")
for(i in 1:Chain_count){
  par(new=T)
  contour(openMP_kde[[i]], xlim=c(-0.5,1.5), ylim=c(1,3), col=i)
}
par(new=T)
contour(full_c, xlim=c(-0.5,1.5), ylim=c(1,3), col="black")


#for the non-parametric algorithm, the variance of the parameters
#is quite big relative to the other 2 algorithm
# BUT this is just by definition of the algorithm and also for our relatively small number of iterations 20000;


####data ellipses
# library(car)
# dataEllipse(test_MCMC_c$Result[(0.5*total_iterations):total_iterations,1],
#               test_MCMC_c$Result[(0.5*total_iterations):total_iterations,2], levels=0.9,
#             plot.points = FALSE, col="black", xlim=c(-0.5,1), ylim=c(1,3))
# par(new=T)
# dataEllipse(test_openMP$Result[(0.2*total_iterations):total_iterations,1],
#             test_openMP$Result[(0.2*total_iterations):total_iterations,2], levels=0.9,
#             plot.points = FALSE, col="red", xlim=c(-0.5,1), ylim=c(1,3))
# par(new=T)
# dataEllipse(test_nonparametric_c[,1], test_nonparametric_c[,2], levels=0.9,
#             plot.points = FALSE, col="blue", xlim=c(-0.5,1), ylim=c(1,3))
# par(new=T)
# dataEllipse(test_parametric[,1], test_parametric[,2], levels=0.9,
#             plot.points = FALSE, col="yellow", xlim=c(-0.5,1), ylim=c(1,3))
# par(new=T)
# dataEllipse(test_semiparametric[,1], test_semiparametric[,2], levels=0.9,
#             plot.points = FALSE, col="pink", xlim=c(-0.5,1), ylim=c(1,3))
