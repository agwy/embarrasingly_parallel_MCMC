
# This file contains wrappers for the C functions that perform a random walk MCMC algorithm that target 
# the posterior distributions of the probit and logit models.

#' GPU_probit_MCMC_C
#'
#' @export GPU_probit_MCMC_C
#' @param Observations  - A vector containing the observations.
#' @param Beta          - A vector containing the initial beta vector for the MCMC algorithm.
#' @param Data_Matrix   - The TRANSPOSED design matrix.
#' @param Iterations    - An integer indicating the number of iterations for the MCMC algorithm.
#' @param proposal_sd   - The standard deviation for the spherical diagonal normal proposals.
#' @return Generates a matrix containing in column j the j-th step in a random walk Metropolis-Hastings algorithm
#'         that targets the posterior distribution of the parameters of a Probit model given realisations of a
#'         design matrix and observations. The proposal distribution is a spherical diagonal normal distribution.
#'         This is a GPU implementation that uses manually coded (parallelized) matrix multiplications. 
#'         Computations for the error function are also parallelized.
GPU_probit_MCMC_C <- function(Observations,Beta,Data_Matrix,Iterations,Proposal_sd) {
  if(!all.equal(dim(Data_Matrix),c(length(Beta),length(Observations))))
    stop("Please Have data matrix in pxn format where length(Observations) = n and length(Beta) = p")
  
  ans <- .C("GPU_probit_MCMC", 
            as.integer(Observations), 
            as.single(Beta), 
            as.single(Data_Matrix),
            Chain_Output = as.single(rep(0,(length(Beta)*Iterations))),
            Acceptance_rate = as.single(0),
            as.integer(length(Observations)),
            as.integer(length(Beta)),
            as.integer(Iterations),
            as.single(Proposal_sd)
            )
  
  return(list(Chain_Output = matrix(ans$Chain_Output,nrow=length(Beta)),
              Acceptance_rate = ans$Acceptance_rate))
}


#' GPU_logit_MCMC_C
#'
#' @export GPU_logit_MCMC_C
#' @param Observations  - A vector containing the observations.
#' @param Beta          - A vector containing the initial beta vector for the MCMC algorithm.
#' @param Data_Matrix   - The TRANSPOSED design matrix.
#' @param Iterations    - An integer indicating the number of iterations for the MCMC algorithm.
#' @param proposal_sd   - The standard deviation for the spherical diagonal normal proposals.
#' @return Generates a matrix containing in column j the j-th step in a random walk Metropolis-Hastings algorithm
#'         that targets the posterior distribution of the parameters of a logit model given realisations of a
#'         design matrix and observations. The proposal distribution is a spherical diagonal normal distribution.
#'         This is a GPU implementation that uses manually coded (parallelized) matrix multiplications. 
#'         Computations the logistic function are also parallelized.
GPU_logit_MCMC_C <- function(Observations,Beta,Data_Matrix,Iterations,Proposal_sd) {
  if(!all.equal(dim(Data_Matrix),c(length(Beta),length(Observations))))
    stop("Please Have data matrix in pxn format where length(Observations) = n and length(Beta) = p")
  
  ans <- .C("GPU_logit_MCMC", 
            as.integer(Observations), 
            as.single(Beta), 
            as.single(Data_Matrix),
            Chain_Output = as.single(rep(0,(length(Beta)*Iterations))),
            Acceptance_rate = as.single(0),
            as.integer(length(Observations)),
            as.integer(length(Beta)),
            as.integer(Iterations),
            as.single(Proposal_sd)
  )
  
  return(list(Chain_Output = matrix(ans$Chain_Output,nrow=length(Beta)),
              Acceptance_rate = ans$Acceptance_rate))
}

#' GPU_logit_MCMC_cuBLAS
#'
#' @export GPU_logit_MCMC_cuBLAS
#' @param Observations  - A vector containing the observations.
#' @param Beta          - A vector containing the initial beta vector for the MCMC algorithm.
#' @param Data_Matrix   - The design matrix (NOT TRANSPOSED).
#' @param Iterations    - An integer indicating the number of iterations for the MCMC algorithm.
#' @param proposal_sd   - The standard deviation for the spherical diagonal normal proposals.
#' @return Generates a matrix containing in column j the j-th step in a random walk Metropolis-Hastings algorithm
#'         that targets the posterior distribution of the parameters of a logit model given realisations of a
#'         design matrix and observations. The proposal distribution is a spherical diagonal normal distribution.
#'         This is a GPU implementation that uses the cuBLAS library to perform matrix multiplications.
#'         Computations the logistic function are also parallelized.
GPU_logit_MCMC_cuBLAS <- function(Observations,Beta,Data_Matrix,Iterations,Proposal_sd) {
  # NOTE: Data_Matrix is not transposed this time:
  if(!all.equal(dim(Data_Matrix),c(length(Observations),length(Beta))))
    stop("Please Have data matrix in pxn format where length(Observations) = n and length(Beta) = p")
  
  ans <- .C("GPU_logit_MCMC_cuBLAS", 
            as.integer(Observations), 
            as.single(Beta), 
            as.single(Data_Matrix),
            Chain_Output = as.single(rep(0,(length(Beta)*Iterations))),
            Acceptance_rate = as.single(0),
            as.integer(length(Observations)),
            as.integer(length(Beta)),
            as.integer(Iterations),
            as.single(Proposal_sd)
  )
  
  return(list(Chain_Output = matrix(ans$Chain_Output,nrow=length(Beta)),
              Acceptance_rate = ans$Acceptance_rate))
}
