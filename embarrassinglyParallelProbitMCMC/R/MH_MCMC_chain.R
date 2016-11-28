MH_MCMC_chain <- function(Iterations,target_density, proposal_sd,inital_value,...){
  
   accepted <- rep(0,Iterations)
   d <- dim(inital_value)[1]
   
   output <- matrix(rep(NA,Iterations*d),nrow=d)
   
  output[,1] <- inital_value
  old_logdensity = target_density(beta=output[,1],...) #
  prop_logdensity = 0 #
  proposed_value <- NA
  for(j in 2:Iterations){
    proposed_value <- output[,j-1] + rnorm(d,0,proposal_sd)
    prop_logdensity = target_density(beta=proposed_value,...) #
    
    if(
      exp(prop_logdensity - old_logdensity) 
      > runif(1)){
        output[,j] <- proposed_value
        accepted[j] <- 1
        old_logdensity = prop_logdensity #
    }else{
      output[,j] <- output[,j-1]
    }
  }
  print(mean(accepted))
  return(output)
}











