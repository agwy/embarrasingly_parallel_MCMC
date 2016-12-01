# This file contains functions to simulate observations and parameters of the probit and logit models, as well as
# computing the likelihood function and the posterior of the parameters given the design matrix and observations.

library(mvtnorm)

# Auxiliary functions for the PROBIT model:


#' sim_probit
#'
#' @export sim_probit
#' @param n         - An integer specifying the number of observations.
#' @param dimension - An integer specifying the dimension of the beta vector.
#' @return Generates values for the parameters, design matrix and observations of a probit model.
#'         Uses independent random samples for the beta parameters and the design matrix.
sim_probit <- function(n,dimension){

  beta <- rnorm(dimension,0,1)
  design_mat <- rmvnorm(n,mean=rep(0,dimension))
  obs <- runif(n) < pnorm(design_mat %*% beta)

  return(list(obs=obs,beta=beta,design_mat=design_mat))
}

#' probit_den
#'
#' @export probit_den
#' @param observations  - A vector of length N. The observations of the probit model, i.e. usually the y_i.
#' @param design_matrix - A N x P matrix. The design matrix of the probit model, i.e. usually the X-matrix.
#' @param beta          - A vector of length P. The parameters of the probit model, i.e. usually the beta-vector.
#' @param to_log        - Boolean. If True, returns the log-likelihood. If False, returns the likelihood.
#' @return The (log-)density function for the probit model of the obervations given the paramaters beta.
probit_den <- function(observations, beta,design_mat,to_log=T){
  p_vals <- pnorm(design_mat %*% beta)
  if(to_log){
    return(sum( log(p_vals[observations]))+ sum(log((1-p_vals[!observations])) ))
  }else{
    return(prod(p_vals[observations])*prod(1-p_vals[!observations]))
  }
}

#'augmented_density_probit
#'
#' @export augmented_density_probit
#' @return Computes the reweighted log-posterior distribution of the parameters of the probit model given
#'         the design matrix and observations. Assumes independent normal priors on the beta vectors.
#'         The reweighting refers to the procedure in (Neiswanger et al),
#'         using an unnormalized prior p(beta)^(1/Chain_count) for each subsetted data-instance.

augmented_density_probit <- function(Chain_count=1,...){
  return(sum(dnorm(x = list(...)$beta,mean = 0,sd = 1,log = T ))*(1/Chain_count) + probit_den(...))
}

## Auxiliary functions for the logit model:

#'sim_logit
#'
#' @export sim_logit
#' @param n         - An integer specifying the number of observations.
#' @param dimension - An integer specifying the dimension of the beta vector.
#' @return Generates values for the parameters, design matrix and observations of a logit model.
#'         Uses independent random samples for the beta parameters and the design matrix.
sim_logit <- function(n,dimension){
  beta <- rnorm(dimension,0,1)
  design_mat <- rmvnorm(n,mean=rep(0,dimension))
  obs <- runif(n) < plogis(design_mat %*% beta)

  return(list(obs=obs,beta=beta,design_mat=design_mat))
}

#'logit_den
#'
#' @export logit_den
#' @param observations  - A vector of length N. The observations of the logit model, i.e. usually the y_i.
#' @param design_matrix - A N x P matrix. The design matrix of the logit model, i.e. usually the X-matrix.
#' @param beta          - A vector of length P. The parameters of the logit model, i.e. usually the beta-vector.
#' @param to_log        - Boolean. If True, returns the log-likelihood. If False, returns the likelihood.
#' @return (log-)The density function for the logistic regression model of the obervations given the paramaters beta.
logit_den <- function(observations, beta, design_mat, to_log =T){
  p_vals <- plogis(design_mat %*% beta)
  if(to_log){
    return(sum( log(p_vals[observations]))+ sum(log((1-p_vals[!observations])) ))
  }else{
    return(prod(p_vals[observations])*prod(1-p_vals[!observations]))
  }
}

#'augmented_density
#'
#' @export augmented_density
#' @return Computes the reweighted log-posterior distribution of the parameters of the logit model given
#'         the design matrix and observations. Assumes independent normal priors on the beta vectors.
#'         The reweighting refers to the procedure in (Neiswanger et al),
#'         using an unnormalized prior p(beta)^(1/Chain_count) for each subsetted data-instance.
augmented_density <- function(Chain_count=1,...){
  return(sum(dnorm(x = list(...)$beta,mean = 0,sd = 1,log = T ))*(1/Chain_count) + logit_den(...))
}
