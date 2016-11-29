#Parametric density product
#burn?

parametric_implementation = function(chain_store, burnin){
  total_iter <- dim(chain_store[[1]])[1]
  M <- length(chain_store)
  d <- dim(chain_store[[1]])[2]
  subset_precision = list()
  subset_mean = list()
  
  for(i in 1:M){
  subset_precision[[i]] = solve(cov(chain_store[[i]][burnin:total_iter,]))
  subset_mean[[i]] = colMeans(chain_store[[i]][burnin:total_iter,])
  }
  
  cov = matrix(0, ncol=d, nrow=d)
  mean = numeric(d)
  
  for(i in 1:M){
    cov = cov + subset_precision[[i]] 
    mean = mean + t(subset_precision[[i]]%*%subset_mean[[i]])
  }
  cov = solve(cov)
  mean = cov %*% t(mean)
  
  theta_out  <- matrix(rep(0,d*(total_iter-burnin)),ncol=d)##store result here
  for(i in 1:(total_iter-burnin)){
    theta_out[i,] <- rmvnorm(n=1, mean=mean, sigma = cov)
  }
  return(theta_out)
}