GPU_MCMC_C <- function(Observations,Beta,Data_Matrix,Iterations,Proposal_sd) {
  if(!all.equal(dim(Data_Matrix),c(length(Beta),length(Observations))))
    stop("Please Have data matrix in pxn format where length(Observations) = n and length(Beta) = p")
  
  ans <- .C("GPU_MCMC", 
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

