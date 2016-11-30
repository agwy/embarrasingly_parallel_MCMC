#Semi-parametric implementation

#Asymptotically Exact Sampling
require(mvtnorm)

#Assume chains in chain_store are of dimension rxn

Weight_cal_SemiParametric <- function(t_dot, chain_store,Gaus_Bandwidth,mu_M,cov_M,subset_mean,subset_cov){
  M <- length(chain_store)

  theta_dot <- sapply(1:length(t_dot),function(x) chain_store[[x]][t_dot[x],])
  theta_bar <- rowMeans(theta_dot)

  bigW_dot <- sum(dmvnorm(x = t(theta_dot),mean = theta_bar,diag(Gaus_Bandwidth^2,length(theta_bar)),log = T)) +
    dmvnorm(x = theta_bar,mean = mu_M,cov_M + diag(Gaus_Bandwidth/M,length(theta_bar)) ,log = T)

  denomin <- sapply(1:M, function(y)
    dmvnorm(x = theta_dot[,y] ,mean = subset_mean[[y]],subset_cov[[y]],log = T)
    )

  return(bigW_dot - sum(denomin) )
}


sample_theta_SemiParametric <- function(t_dot, chain_store,Gaus_Bandwidth,mu_M,precision_M ){
  M <- length(chain_store)
  theta_dot <- sapply(1:length(t_dot),function(x) chain_store[[x]][t_dot[x],])
  theta_bar <- rowMeans(theta_dot)
  cov_t_dot <- solve( diag(M/Gaus_Bandwidth,length(theta_bar)) +  precision_M)
  mu_t_dot <- cov_t_dot %*% (diag(M/Gaus_Bandwidth,length(theta_bar)) %*% theta_bar + precision_M %*% mu_M )

  return(rmvnorm(n = 1,mean = mu_t_dot,sigma = cov_t_dot))
}

Parametric_mu_cov <- function(chain_store){
  total_iter <- dim(chain_store[[1]])[1]
  M <- length(chain_store)
  d <- dim(chain_store[[1]])[2]
  subset_precision <-  subset_mean <- subset_cov <-  list()

  for(i in 1:M){
    subset_cov[[i]] = cov(chain_store[[i]])
    subset_precision[[i]] = solve(subset_cov[[i]])
    subset_mean[[i]] = colMeans(chain_store[[i]])
  }

  cov = matrix(0, ncol=d, nrow=d)
  mean = numeric(d)

  for(i in 1:M){
    cov = cov + subset_precision[[i]]
    mean = mean + t(subset_precision[[i]]%*%subset_mean[[i]])
  }
  cov = solve(cov)
  mean = cov %*% t(mean)
  return(list(mean=mean,cov=cov,subset_mean = subset_mean,subset_cov=subset_cov))
}


#'semiparametric_implementation
#'
#' @export Semiparametric_implementation
#' @param chain_store - A list of M matrices, where M is the number of subsets.
#'                      Each matrix contains the steps of the MH chain on each subset of data.
#'                      The dimension of each matrix is Iterations x d, where d is the number of parameters.
#' @param burnin - A number specifying the burnin period, i.e. how many iterations will be discarded.
#'                 Usually taken as 0.2*Iterations.
#' @return A matrix containing all the steps of the combined MH chain which is of dimension (Iterations - burnin)xd.
Semiparametric_implementation <- function(chain_store, burnin, Verbose = TRUE){

  d <- dim(chain_store[[1]])[2]
  total_iter <- dim(chain_store[[1]])[1]
  M <- length(chain_store)

  ##First adjust for burnin
  for(i in 1:M){
    chain_store[[i]] = chain_store[[i]][burnin:total_iter,]
  }
  total_iter <- dim(chain_store[[1]])[1]

  #Parametric estimates - calculate now
  parametric_ests <-  Parametric_mu_cov(chain_store)
  mu_M <- parametric_ests$mean
  cov_M <- parametric_ests$cov
  precision_M <- solve(parametric_ests$cov)
  subset_mean <- parametric_ests$subset_mean
  subset_cov <- parametric_ests$subset_cov
  rm(parametric_ests)

  t_dot <- sample(1:total_iter,M,replace = T)

  #For storing Output
  theta_out  <- matrix(rep(0,d*total_iter),ncol=d)

  for(k in 1:total_iter){
    bandwidth_set <- k^(-1/(4 + d))
    current_weight <- Weight_cal_SemiParametric(t_dot = t_dot,Gaus_Bandwidth =bandwidth_set,chain_store = chain_store,
                                               mu_M=mu_M,cov_M = cov_M,
                                               subset_mean = subset_mean,subset_cov=subset_cov)
    for(j in 1:M){
      c_dot <- t_dot
      c_dot[j] <- sample(1:total_iter,1)
      u <- runif(1)
      new_weight <-Weight_cal_SemiParametric(t_dot = c_dot,Gaus_Bandwidth =bandwidth_set,chain_store = chain_store,
                                             mu_M=mu_M,cov_M = cov_M,
                                             subset_mean = subset_mean,subset_cov=subset_cov)

      if( u <  exp( new_weight - current_weight)){
        t_dot <- c_dot
        current_weight <- new_weight
      }

    }
    if( (k %% floor(total_iter* 0.1) == 0)  & Verbose ) print(paste("Iteration: ", round(k/total_iter,digits = 3)))

    theta_out[k,] <- sample_theta_SemiParametric(t_dot,chain_store,bandwidth_set,mu_M,precision_M)
  }

  #Faster Implementation -- seeems very serial, what can be vectorised?

  return(theta_out)
}





