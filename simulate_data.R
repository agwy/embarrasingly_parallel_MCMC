#gaussian mixture model
#generate data

library(MASS)

# L is the number of components
# D is the dimensionality of the gaussians
# N is the number of observations

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
  
  res = list(mvn_data = mvn_data, mean = mean, sigma2 = sigma2, n = n)
  return(res)
}


set.seed(9)
L=10
D=2
N=50000
data_mvn = generate_data(L, D, N)
plot(data_mvn[[1]], ylim=c(0,L), xlim=c(0,L))
