#probit model 
sim_probit <- function(n,dimension){
  beta <- rnorm(dimension,0,1)
  design_mat <- rmvnorm(n,mean=rep(0,dimension))
  
  obs <- runif(n) < pnorm(design_mat %*% beta)
  
  return(list(obs=obs,beta=beta,design_mat=design_mat))
}

probit_den <- function(observations, beta,design_mat,to_log=T){
  p_vals <- pnorm(design_mat %*% beta)
  if(to_log){
    return(sum( log(p_vals[observations]))+ sum(log((1-p_vals[!observations])) ))
  }else{
    return(prod(p_vals[observations])*prod(1-p_vals[!observations]))
  }
}

augmented_density <- function(Chain_count=1,...){
  return(sum(dnorm(x = list(...)$beta,mean = 0,sd = 1,log = T ))*(Chain_count) + probit_den(...))
}

