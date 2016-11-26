#MHCMCMC wrapper

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

MCMC_MH_parallel <- function(M, Iterations, Data_Matrix, Obs,Inital_beta,proposal_sd) {
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
            as.double(M*length(Obs)), #
            acceptance_rate = as.double(1:M),
            Result = as.double(1:(M*length(Inital_beta)*(Iterations+1)))
  )
  print("HEllo world")
  return(list(Result = matrix(ans$Result,ncol=length(Inital_beta), byrow=TRUE),
              Acceptance_rate = ans$acceptance_rate))
}

########################################
# #Test of augmented density:
#
# #dyn.load("c_code/a.so")
#
# augmented_density_c <- function( M, num_data, num_param, design_matrix, obs, beta, tmp, res){
#   ans <- .C("augmented_density",
#             as.integer(M),
#             as.integer(length(obs)),
#             as.integer(length(beta)),
#             as.double(design_matrix),
#             as.integer(obs),
#             as.double(beta),
#             as.double(tmp),
#             Res = as.double(res)
#             )
#   return(ans$Res)
# }
#
# library(Rcpp)
# library(mvtnorm)
# library(parallel)
#
#
# source("probit_funcs.R")
# source("MH_MCMC_chain.R")
#
# probit_dimension <- 50
# obs_count <- 500000
#
# simulated_probit_data <- sim_probit(obs_count,probit_dimension)
#
# glm_test <- glm(simulated_probit_data$obs~simulated_probit_data$design_mat + 0,family = binomial(link="probit"))
#
# probit_den(observations = simulated_probit_data$obs,
#            beta = glm_test$coefficients,
#            design_mat = simulated_probit_data$design_mat)
#
# augmented_density(observations = simulated_probit_data$obs,
#                   beta = glm_test$coefficients,
#                   design_mat = simulated_probit_data$design_mat,
#                   to_log=T)
#
# tmp = rep(0.0, times=obs_count)
# res = 0.0;
# augmented_density_c(1, obs_count,  probit_dimension, simulated_probit_data$design_mat, simulated_probit_data$obs, glm_test$coefficients, tmp, res)
#



