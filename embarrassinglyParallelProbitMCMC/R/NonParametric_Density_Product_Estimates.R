#Asymptotically Exact Sampling
require(mvtnorm)

#Assume chains in chain_store are of dimension rxn

Weight_cal <- function(t_dot, chain_store,Gaus_Bandwidth){
  theta_dot <- sapply(1:length(t_dot),function(x) chain_store[[x]][t_dot[x],])
  theta_bar <- rowMeans(theta_dot)

  w_dot <- sum(dmvnorm(x = t(theta_dot),mean = theta_bar,diag(Gaus_Bandwidth^2,length(theta_bar)),log = T))
  return(w_dot)
}


sample_theta <- function(t_dot, chain_store,Gaus_Bandwidth ){
  M <- length(chain_store)
  theta_dot <- sapply(1:length(t_dot),function(x) chain_store[[x]][t_dot[x],])
  theta_bar <- rowMeans(theta_dot)
  return(rmvnorm(n = 1,mean = theta_bar,sigma = diag((Gaus_Bandwidth^2)/M,length(theta_bar))))
}

#'nonparametric_implementation
#'
#' @export nonparametric_implementation
#' @param chain_store - A list of M matrices, where M is the number of subsets.
#'                      Each matrix contains the steps of the MH chain on each subset of data.
#'                      The dimension of each matrix is Iterations x d, where d is the number of parameters.
#' @param burnin - A number specifying the burnin period, i.e. how many iterations will be discarded.
#'                 Usually taken as 0.2*Iterations.
#' @return A matrix containing all the steps of the combined MH chain which is of dimension (Iterations - burnin)xd.
nonparametric_implementation <- function(chain_store, burnin, Verbose = TRUE){

  d <- dim(chain_store[[1]])[2]
  total_iter <- dim(chain_store[[1]])[1]
  M <- length(chain_store)

  for(i in 1:M){
    chain_store[[i]] = chain_store[[i]][burnin:total_iter,]
  }

  total_iter <- dim(chain_store[[1]])[1]


  t_dot <- sample(1:total_iter,M,replace = T)
  #print(t_dot)

  #For storing Output
  theta_out  <- matrix(rep(0,d*total_iter),ncol=d)

  for(k in 1:total_iter){
    bandwidth_set <- k^(-1/(4 + d))
    old_weight <- Weight_cal(t_dot = t_dot,Gaus_Bandwidth =bandwidth_set,chain_store = chain_store )
    for(j in 1:M){
      c_dot <- t_dot
      c_dot[j] <- sample(1:total_iter,1)
      u <- runif(1)
      new_weight <-Weight_cal(t_dot = c_dot,Gaus_Bandwidth =bandwidth_set,chain_store = chain_store )
      if( u <  exp( new_weight- old_weight)
      ) {
        t_dot <- c_dot
        old_weight <- new_weight
      }
    }
    if( (k %% floor(total_iter* 0.1) == 0)  & Verbose ) print(paste("Iteration: ", round(k/total_iter,digits = 3)))

    theta_out[k,] <- sample_theta(t_dot,chain_store,bandwidth_set)
  }

  #Faster Implementation -- seeems very serial, what can be vectorised?

  return(theta_out)
}





