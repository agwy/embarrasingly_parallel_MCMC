#Parametric density product

#'parametric_implementation
#'
#' @export parametric_implementation
#' @param chain_store - A list of M matrices, where M is the number of subsets.
#'                      Each matrix contains the steps of the MH chain on each subset of data.
#'                      The dimension of each matrix is Iterations x d, where d is the number of parameters.
#' @param burnin - A number specifying the burnin period, i.e. how many iterations will be discarded.
#'                 Usually taken as 0.2*Iterations.
#' @return A matrix containing all the steps of the combined MH chain which is of dimension (Iterations - burnin)xd.
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
