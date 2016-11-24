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
  
  n = ceiling(N*probs) # number of observations for each gaussian
  print(n)
  
  sigma2 = numeric(L)
  mean = matrix(NA, ncol = D, nrow = L)
  
  for(i in 1:L){
    sigma2[i] = runif(1,0,0.01)
    mean[i,] = runif(D,0,10)
  }
  
  #generate first gaussian outside of the loop
  mvn_data = mvrnorm(n[1], mean[1,], diag(sigma2[1], D))

  for(i in 2:L){
    mvn_data = rbind(mvn_data, mvrnorm(n[i], mean[i,], diag(sigma2[i], D)))
  }
  
  res = list(mvn_data = mvn_data, mean = mean, sigma2 = sigma2, n = n, probs=probs)
  return(res)
}


set.seed(9)
L=10
D=2
N=1000
simulated_data = generate_data(L, D, N)
plot(simulated_data[[1]], ylim=c(0,L), xlim=c(0,L))


####log likelihood function####


#' @param the output from the generate_data function
#' @param proposal is a vector of probs, means, variances
#' 
log_likelihood = function(X, proposal){
  #get all the variables needed
  D = ncol(X)
  n = nrow(X)
  L = length(proposal)/(2+D)
  
  probs = proposal[1:L]
  mean = matrix(proposal[(L+1): (L*D)], nrow=L, byrow=T)
  sigma = proposal[(D*L+1):length(proposal)]
  
  tmp = numeric(L)
  obs_prob = numeric(n)
  
  for(i in 1:n){
    for(j in 1:L)
    tmp[j]=probs[j]*dmvnorm(X[i,], mean[j,], diag(sigma[j], D))
  }
  
  obs_prob = sum(tmp)
  
  log_lik = sum(log(obs_prob))
  return(log_lik)
}

log_prior = function(proposal){
  
}


log_target = function(X_m, proposal, M=1){
  res = log_likelihood(X_m, proposal) + log_prior(proposal)*(1/M)
  return(res)
}
