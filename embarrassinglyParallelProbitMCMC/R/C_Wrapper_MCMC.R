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
# source("logit_funcs.R")
# source("MH_MCMC_chain.R")
#
# logit_dimension <- 50
# obs_count <- 500000
#
# simulated_logit_data <- sim_logit(obs_count,logit_dimension)
#
# glm_test <- glm(simulated_logit_data$obs~simulated_logit_data$design_mat + 0,family = binomial(link="logit"))
#
# logit_den(observations = simulated_logit_data$obs,
#            beta = glm_test$coefficients,
#            design_mat = simulated_logit_data$design_mat)
#
# augmented_density(observations = simulated_logit_data$obs,
#                   beta = glm_test$coefficients,
#                   design_mat = simulated_logit_data$design_mat,
#                   to_log=T)
#
# tmp = rep(0.0, times=obs_count)
# res = 0.0;
# augmented_density_c(1, obs_count,  logit_dimension, simulated_logit_data$design_mat, simulated_logit_data$obs, glm_test$coefficients, tmp, res)
#



