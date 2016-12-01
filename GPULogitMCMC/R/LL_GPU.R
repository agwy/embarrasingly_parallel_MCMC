
#' LL_probit_GPU
#'
#' @export LL_probit_GPU
#' @param Observations  - A vector containing the observations.
#' @param Beta          - A vector containing a parameter vector for the probit model.
#' @param Data_Matrix   - The design matrix.
#' @return A wrapper for C code that computes the probit density function. 
#'         It concerns a GPU implementation that parallelises the matrix multiplications (manually implemented) and 
#'         the computation of the probit density.
LL_probit_GPU <- function(Observations, Beta, Data_Matrix) {
  if(!all.equal(dim(Data_Matrix),c(length(Beta),length(Observations))))
    stop("Please have the (transposed) data matrix in p x n format where length(Observations) = n and length(Beta) = p.")
  
  ans <- .C("GPU_Probit_LL", 
            as.integer(Observations), 
            as.single(Beta), 
            as.single(Data_Matrix),
            Log_Likelihood= as.single(0),
            as.integer(length(Observations)),
            as.integer(length(Beta)))
  return(ans$Log_Likelihood)
}

#' LL_logit_GPU
#'
#' @export LL_logit_GPU
#' @param Observations  - A vector containing the observations.
#' @param Beta          - A vector containing a parameter vector for the probit model.
#' @param Data_Matrix   - The design matrix.
#' @return A wrapper for C code that computes the logit density function. 
#'         It concerns a GPU implementation that parallelises the matrix multiplications (manually implemented) and 
#'         the computation of the logit density.
LL_logit_GPU <- function(Observations, Beta, Data_Matrix) {
  if(!all.equal(dim(Data_Matrix),c(length(Beta),length(Observations))))
    stop("Please have the (transposed) data matrix in p x n format where length(Observations) = n and length(Beta) = p.")
  
  ans <- .C("GPU_logit_LL", 
            as.integer(Observations), 
            as.single(Beta), 
            as.single(Data_Matrix),
            Log_Likelihood= as.single(0),
            as.integer(length(Observations)),
            as.integer(length(Beta)))
  return(ans$Log_Likelihood)
}

