#####generate data###
#####################

library(MASS)
library(birdring)

#'
#' @param L:  the number of components
#' @param D:  the dimensionality of the Gaussians
#' @param N:  total number of observations
#' 
#' @return A list [[1]] a matrix with D number of columns with the observations for the corresponding dimension;
#'                [[2]] the mean matrix used for the simulation;
#'                [[3]] the sigma vector used for the simulation;
#'                [[4]] the number of observations for each Gaussion
#'                [[5]] the vector of probabilities of simulating from each Gaussion.

generate_data = function(L=10, D=2, N=50000){
  probs = runif(L,0,1)
  probs = probs/sum(probs) #standardize
  
  allocations = sample(seq_len(L), N, replace = TRUE) # number of observations for each gaussian
  
  sigma2 = numeric(L)
  mean = matrix(NA, ncol = D, nrow = L)
  
  for(i in 1:L){
    sigma2[i] = runif(1,0,0.01)
    mean[i,] = runif(D,0,10)
  }
  
  #generate first gaussian outside of the loop
  mvn_data = mvrnorm(1, mean[allocations[1],], diag(sigma2[allocations[1]], D))
  
  for(i in allocations[-1]){
    mvn_data = rbind(mvn_data, mvrnorm(1, mean[i,], diag(sigma2[i], D)))
  }
  
  res = list(mvn_data = mvn_data, mean = mean, sigma2 = sigma2, probs=probs)
  return(res)
}

split_data = function(X, M){
  N = nrow(X)
  num_per_subset = N / M
  if(is.integer(num_per_subset)){
    stop("M does not divide the number of data points!")
  }
  subsetted_data = list()
  for(i in seq_len(M)){
    subsetted_data[[i]] = X[(i-1) * num_per_subset + seq_len(num_per_subset),  ]
  }
  return(subsetted_data)
}


# set.seed(13)
# L=10
# D=2
# N=1000
# simulated_data = generate_data(L, D, N)
# plot(simulated_data[[1]], ylim=c(0,L), xlim=c(0,L))
# subsetted_data = split_data(simulated_data[[1]], 10)
# plot(subsetted_data[[2]], ylim=c(0,L), xlim=c(0,L))

####log likelihood function####


#' @param the output from the generate_data function
#' @param proposal is a vector of probs, means, variances
#' 
log_likelihood = function(X, proposal){
  #get all the variables needed
  D = ncol(X)
  n = nrow(X)
  L = length(proposal)/ (D + 2)
  
  probs = proposal[1:L]
  mean = matrix(proposal[(L+1): ((L*D)+ L)], nrow=L, byrow=T)
  sigma = proposal[(D*L + L +1):length(proposal)]
  
  tmp = numeric(L)
  log_lik = 0
  
  for(i in 1:n){
    for(j in 1:L){
      tmp[j]=probs[j]*dmvnorm(X[i,], mean[j,], diag(sigma[j], D))
    }
    log_lik = log_lik + log(sum(tmp))
  }
  
  return(log_lik)
}

#log_likelihood(simulated_data[[1]], c(.5,.5,7,1,5,9,.01,.01))


log_prior = function(proposal, D){
  L = length(proposal)/(2+D)
  
  probs = proposal[1:L]
  mean = matrix(proposal[(L+1): (L*D)], nrow=L, byrow=T)
  sigma = proposal[(D*L+1):length(proposal)]
  
  # log_prob = 0
  # for(i in seq_len(L)){
  #   log_prob = log_prob + dmvnorm(mean[i,], mean = rep(5, times = D), diag(1, D), log.p = TRUE)
  # }
  
  return(0) # Unifrom prior, yay!
}


log_target = function(X_m, proposal, M=1){
  D = ncol(X_m)
  res = log_likelihood(X_m, proposal) + log_prior(proposal, D)*(1/M)
  return(res)
}


