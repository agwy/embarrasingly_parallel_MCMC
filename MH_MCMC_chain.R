MH_MCMC_chain <- function(Iterations,target_density, proposal_sd,inital_value,...){
  
   accepted <- rep(0,Iterations)
   d <- dim(inital_value)[1]
   
   output <- matrix(rep(NA,Iterations*d),nrow=d)
   
  output[,1] <- inital_value
  proposed_value <- NA
  for(j in 2:Iterations){
    proposed_value <- output[,j-1] + rnorm(d,0,proposal_sd)
    
    if(
      exp(sum(target_density(beta=proposed_value,...)) - sum(target_density(beta=output[,j-1],...)))
      > runif(1)){
        output[,j] <- proposed_value
        accepted[j] <- 1
    }else{
      output[,j] <- output[,j-1]
    }
  }
  print(mean(accepted))
  return(output)
}











