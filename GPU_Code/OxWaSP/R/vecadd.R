LL_GPU <- function(Observations,Beta,Data_Matrix) {
  if(!all.equal(dim(Data_Matrix),c(length(Beta),length(Observations))))
    stop("Please Have data matrix in pxn format where length(Observations) = n and length(Beta) = p")
  
  ans <- .C("GPU_Probit_LL", 
            as.integer(Observations), 
            as.single(Beta), 
            as.single(Data_Matrix),
            Log_Likelihood= as.single(0),
            as.integer(length(Observations)),
            as.integer(length(Beta)))
  return(ans$Log_Likelihood)
}

