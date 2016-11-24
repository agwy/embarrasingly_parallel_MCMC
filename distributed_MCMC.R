library(parallel)
source("simulate_data.R")
source("NonParametric_Density_Product_Estimates.R")

set.seed(15)
num_cores = 4
L=2
D=2
N=1000 # Number of observations
M = 2 # Number of subsets; should divide N

# Generate data:
simulated_data = generate_data(L, D, N)
plot(simulated_data[[1]], ylim=c(0,10), xlim=c(0,10))
subsetted_data = split_data(simulated_data[[1]], M)
plot(subsetted_data[[2]], ylim=c(0,10), xlim=c(0,10))

list_of_log_targets = list()
for (i in seq_len(M)){
  list_of_log_targets[[i]] = function(proposal){
    return(log_target(subsetted_data[[i]], proposal, M))
  }
}

# First load mcmc algorithm from below
mclapply(list_of_log_targets, mcmc_algorithm, mc.cores = num_cores) # Needs a valid mcmc algorithm

mcmc_algorithm(list_of_log_targets[[1]])

# For adaptive MCMC:
# mclapply(list(p.log1, p.log2), adapt_mcmc_algorithm, mc.cores = 2)

########################
## SPHERICAL RW MCMC  ##
########################

spherical_random_walk_mcmc = function(log_target, num_its, init_value, sigma, D, L){
  #'
  #' @param log_target:     The logarithm of an unnormalized density function for the target distribution.
  #' @param num_its:    An integer, the number of iterations to run the mcmc algorithm for.
  #' @param init_value: A vector containing the initial position for the random walk.
  #' @param sigma:      A positive number, the standard deviation for the covariance 
  #'                    of the diagonal spherical normal proposal distribution.
  #'
  #' @return A list of vectors constituting a Metropolis-Hastings random walk targetting 
  #'         the target distribution with spherical normal distributed proposals.
  
  num_dim = length(init_value)
  
  walk = list(init_value) # TODO: initialize length beforehand (speed-up?)? 
  proposal = 0
  
  for(i in seq_len(num_its)){
    proposal = walk[[i]] + rnorm(num_dim, sd = sigma)
    proposal[1:L] = abs(proposal[1:L] / sum(proposal[1:L]))
    log_acc_ratio = log_target(proposal) - log_target(walk[[i]]) # Careful for log(0) when implementing this in C
    
    if(log_acc_ratio > 0 || log_acc_ratio > log(runif(1))){
      walk[[i+1]] = proposal
    } else {
      walk[[i+1]] = walk[[i]]
    }
    
    if(i %% (floor(num_its/10)) == 0){
      print(paste("/r", i/num_its * 100, "%"))
    }
    
  }
  
  walk_matrix = do.call(walk, rbind)
  return(walk_matrix)
}


mcmc_algorithm = function(log_target){
  num_its = 1000
  init_value = c(.4, .6, 5, 5, 5, 6, .1, .1 )
  sigma = 1
  D = D
  L = L
  return(spherical_random_walk_mcmc(log_target, num_its, init_value, sigma, D, L))
}


#####################
##  ADAPTIVE MCMC  ##
#####################

library(adaptMCMC)
adapt_mcmc_algorithm = function(log_target){
  num_its = 10000
  init_value = rep(0, times = 2)
  sigma = 1
  mcmc_object = MCMC(log_target, n = num_its, init = init_value, scale = rep(1, times = length(init_value)), 
                     adapt = TRUE, acc.rate = .234)
  return(mcmc_object$samples)
}

# Test log_target
p.log1 <- function(x) {
  B <- 0.03 # controls 'bananacity'
  -x[1]^2/200 - 1/2*(x[2]+B*x[1]^2-100*B)^2
}
p.log2 <- function(x) {
  B <- 0.05 # controls 'bananacity'
  -x[1]^2/200 - 1/2*(x[2]+B*x[1]^2-100*B)^2
}
