#MH MCMC wrappers

#'MCMC_MH
#'
#' @export MCMC_MH
#' @param M - Number of subsets of the data.
#' @param Iterations - Number of iterations for the Markov chain.
#' @param Data_Matrix - A N x P matrix. The design matrix of the probit model, i.e. usually the X-matrix.
#' @param Obs - A vector of length N. The observations of the logit model.
#' @param Initial_beta - A vector of starting values for the chains, usually taken as a vector of zeroes.
#' @param proposal_sd - A number for the standard deviation used for the proposal. It is the same for all parameters.
#' @return A list of [[1]] a matrix containing all the steps of the MH chain of dimension (Iterations+1) x d,
#'                          where d is the number of parameters, i.e. length of the vector initial_value,
#'                          and the first row is the Initial_beta;
#'                  [[2]] the acceptance rate for the chain.
MCMC_MH <- function(M, Iterations, Data_Matrix, Obs,Inital_beta,proposal_sd) {
  tmp = rep(0, times = length(Obs))
  ans <- .C("MCMC",
            as.integer(Iterations),
            as.integer(length(Obs)),
            as.integer(length(Inital_beta)),
            as.double(Data_Matrix),
            as.integer(Obs),
            as.double(proposal_sd),
            as.double(Inital_beta),
            as.integer(M), #
            as.double(tmp), #
            acceptance_rate = as.double(-1),
            Result = as.double(1:(length(Inital_beta)*(Iterations+1)))
            )
  print("HEllo world")
  return(list(Result = matrix(ans$Result,ncol=length(Inital_beta), byrow=TRUE),
              Acceptance_rate = ans$acceptance_rate))
}


#'MCMC_MH_parallel
#'
#' @export MCMC_MH_parallel
#' @param M - Number of subsets of the data.
#' @param Iterations - Number of iterations for the Markov chain.
#' @param Data_Matrix - A N x P matrix. The design matrix of the probit model, i.e. usually the X-matrix.
#' @param Obs - A vector of length N. The observations of the logit model.
#' @param Initial_beta - A vector of starting values for the chains, usually taken as a vector of zeroes.
#' @param proposal_sd - A number for the standard deviation used for the proposal. It is the same for all parameters.
#' @return A list of [[1]] a matrix containing all the steps of the MH chain of dimension M*(Iterations+1) x d,
#'                          where d is the number of parameters, i.e. length of the vector initial_value,
#'                          and the first row is the Initial_beta;
#'                          The first (Iterations+1) rows contain the MH chain for the first subset, etc.
#'                  [[2]] a vector of acceptance rates for the chains for each subset.
MCMC_MH_parallel <- function(M, Iterations, Data_Matrix, Obs,Inital_beta,proposal_sd) {
  tmp = rep(0, times = length(Obs))
  # The following can perhaps still be optimized. TODO: Make a block of memory available initially.
  # It rearranges the data in an array of subsequent columnmajor-indexed submatrices, subsetted as in Neiswanger.
  design_array = c()
  num_per_subset = length(Obs) / M
  for (m in seq_len(M)){
    design_array = c(design_array, as.double(Data_Matrix[ ((m-1)*num_per_subset + 1) : (m * num_per_subset), ]))
  }
  ans <- .C("openMP",
            as.integer(Iterations),
            as.integer(length(Obs)),
            as.integer(length(Inital_beta)),
            design_array,
            as.integer(Obs),
            as.double(proposal_sd),
            as.double(Inital_beta),
            as.integer(M), #
            as.double(tmp), #
            acceptance_rate = as.double(1:M),
            Result = as.double(1:(M*length(Inital_beta)*(Iterations+1)))
  )
  print("HEllo world")
  return(list(Result = matrix(ans$Result,ncol=length(Inital_beta), byrow=TRUE),
              Acceptance_rate = ans$acceptance_rate))
}



