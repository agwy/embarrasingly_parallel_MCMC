#MHCMCMC wrapper
MCMC_MH <- function(Iterations, Data_Matrix, Obs,Inital_beta,proposal_sd) {
  ans <- .C("MCMC", 
            as.integer(Iterations),
            as.integer(length(Obs)),
            as.integer(length(Inital_beta)),
            as.double(Data_Matrix),
            as.integer(Obs),
            as.double(proposal_sd),
            as.double(Inital_beta),
            acceptance_rate = as.double(1:Iterations),
            Result = as.double(1:(length(Inital_beta)*(Iterations+1)))
            )
  return(list(Result = matrix(ans$Result,ncol=length(Inital_beta), byrow=TRUE),
              Acceptance_rate = ans$acceptance_rate))
}

