#Logistic regression is used to demonstrate the speed of the parallelization#

#at first Probit regression was implemented so there could be some
#commented lines of code related to Probit;

library(Rcpp)
library(mvtnorm)
library(parallel)
library(profr)
library(boot)

library(embarrassinglyParallelProbitMCMC, lib.loc="Packages")

#########################################################################
#### simulate the data #####

logit_dimension <- 50
obs_count <- 4e4

set.seed(15)
simulated_logit_data <- sim_logit(obs_count,logit_dimension)

#########################################################################
#### MCMC approximation ####

total_iterations <- 2e4
proposal_sd <- 0.01 
#for sd 0.03 the acc rate is only 1% for the single chain;
#sd 0.01 is used instead and the acc rate for the single chain is about 12%;
#initial value is taken to be a vector of 0s;

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
# "%*%"                56.60    51.90      56.60     51.90
# "logit_den"          26.44    24.24     108.12     99.14
# "plogis"             21.70    19.90      78.30     71.80
# "!"                   2.44     2.24       2.44      2.24
# "sum"                 0.64     0.59       0.64      0.59
# "MH_MCMC_chain"       0.40     0.37     109.06    100.00
# "-"                   0.20     0.18       0.20      0.18
# "target_density"      0.14     0.13     108.38     99.38
# "runif"               0.14     0.13       0.14      0.13
# "rnorm"               0.12     0.11       0.12      0.11
# "dnorm"               0.10     0.09       0.12      0.11
# "logit"               0.06     0.06       0.10      0.09
# "qlogis"              0.04     0.04       0.04      0.04
# "inv.logit"           0.02     0.02      78.32     71.81
# "list"                0.02     0.02       0.02      0.02
# 
# $by.total
# total.time total.pct self.time self.pct
# "MH_MCMC_chain"      109.06    100.00      0.40     0.37
# "target_density"     108.38     99.38      0.14     0.13
# "logit_den"          108.12     99.14     26.44    24.24
# "inv.logit"           78.32     71.81      0.02     0.02
# "plogis"              78.30     71.80     21.70    19.90
# "%*%"                 56.60     51.90     56.60    51.90
# "!"                    2.44      2.24      2.44     2.24
# "sum"                  0.64      0.59      0.64     0.59
# "-"                    0.20      0.18      0.20     0.18
# "runif"                0.14      0.13      0.14     0.13
# "rnorm"                0.12      0.11      0.12     0.11
# "dnorm"                0.12      0.11      0.10     0.09
# "logit"                0.10      0.09      0.06     0.06
# "qlogis"               0.04      0.04      0.04     0.04
# "list"                 0.02      0.02      0.02     0.02
# 
# $sample.interval
# [1] 0.02
# 
# $sampling.time
# [1] 109.06
# 
# > proc.time() - first_time
# user  system elapsed 
# 109.126   0.010 109.444 
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

############################################################
####Parallel implementation using openMP
############################################################

first_time = proc.time()
test_openMP <- MCMC_MH_parallel(Chain_count, total_iterations, simulated_logit_data$design_mat,
                                simulated_logit_data$obs,rep(0, times=logit_dimension),
                                proposal_sd)
proc.time() - first_time
# > proc.time() - first_time
# user  system elapsed 
# 230.551   0.083  31.553 

# Inspect the first beta for the first chain to visually check convergence:
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
test_nonparametric <- nonparametric_implemetation(test3, burnin=0.2*total_iterations)
proc.time()-first_time
# > proc.time()-first_time
# user  system elapsed 
# 62.555   0.094  62.836 

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
test_nonparametric_c <- nonparametric_implemetation(test_OpenMP_list, burnin=0.2*total_iterations)

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
comb_nonpar = kde2d(test_nonparametric_c[,1], test_nonparametric_c[,2], n=30)

#(4) the combined chain using the parametric algorithm
comb_par = kde2d(test_parametric[,1], test_parametric[,2], n=30)



#plot in a contour plot the full chain and the combined chains
contour(full_c, xlim=c(-0.5,1.5), ylim=c(1,3), col="black")
par(new=T)
contour(comb_nonpar,  xlim=c(-0.5,1.5), ylim=c(1,3), col="blue")
par(new=T)
contour(comb_par, xlim=c(-0.5,1.5), ylim=c(1,3), col="yellow")


#the Nonparametric chain and the openMP subsets; density for beta1 and beta2
contour(comb_nonpar,  xlim=c(-0.5,1.5), ylim=c(1,3), col="blue")
for(i in 1:Chain_count){
  par(new=T)
  contour(openMP_kde[[i]], xlim=c(-0.5,1.5), ylim=c(1,3), col=i)
}
par(new=T)
contour(full_c, xlim=c(-0.5,1.5), ylim=c(1,3), col="black")

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
