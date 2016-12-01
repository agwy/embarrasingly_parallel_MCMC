#'MH_MCMC_chain
#'
#' @export MH_MCMC_chain
#' @param Iterations - Number of iterations for the Markov chain.
#' @param target_density - augmented_density function as defined.
#' @param proposal_sd - A number for the standard deviation used for the proposal. It is the same for all parameters.
#' @param initial_value - A vector of starting values for the chains, usually taken as a vector of zeroes.
#' @return A matrix containing all the steps of the MH chain of dimension d x Iterations,
#'          where d is the number of parameters, i.e. length of the vector initial_value.
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











