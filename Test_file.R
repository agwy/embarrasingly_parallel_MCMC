##Test file for R package
library(mvtnorm)

detach("package:OxWaSP",unload = TRUE)
library(OxWaSP,lib.loc = "Project_4/Packages/")

is.loaded("GPU_Probit_LL")
dyn.load("Project_4/Packages/OxWaSP/libs/OxWaSP.so")
dyn.unload("Project_4/Packages/OxWaSP/libs/OxWaSP.so")
source("OxWaSP/R/vecadd.R")
source("OxWaSP/R/probit_funcs.R")

test_data <- sim_probit(10000,10)

LL_GPU(Observations = test_data$obs,Beta = test_data$beta,Data_Matrix = t(test_data$design_mat))


probit_den(observations = test_data$obs,
             beta = test_data$beta,
             design_mat = test_data$design_mat,to_log = T)

### Testing full semi-MCMC chain on GPU
source("OxWaSP/R/GPU_MCMC.R")
dim(test_data$design_mat)

dyn.unload("Project_4/Packages/OxWaSP/libs/OxWaSP.so")
dyn.load("Project_4/Packages/OxWaSP/libs/OxWaSP.so")
test_GPU_MCMC <-GPU_MCMC_C(
  Observations = test_data$obs,
  Beta = rep(0, 10),
  Data_Matrix = t(test_data$design_mat),
  Iterations = 10000,
  Proposal_sd = 0.1)
test_GPU_MCMC$Acceptance_rate
test_GPU_MCMC$Chain_Output

source("OxWaSP/R/MH_MCMC_chain.R")
test_standard <- MH_MCMC_chain(Iterations = 10000,
                               target_density = probit_den,
                               proposal_sd = 0.1,inital_value = as.matrix(rep(0,10)),
                               observations = test_data$obs,
                               design_mat = test_data$design_mat,to_log = T
                               )

rowMeans(test_GPU_MCMC$Chain_Output)
rowMeans(test_standard)

plot(rowMeans(test_GPU_MCMC$Chain_Output),rowMeans(test_standard))

system.time(test_standard <- MH_MCMC_chain(Iterations = 10000,
                                           target_density = probit_den,
                                           proposal_sd = 0.1,inital_value = as.matrix(rep(0,10)),
                                           observations = test_data$obs,
                                           design_mat = test_data$design_mat,to_log = T
))
system.time(test_GPU_MCMC <-GPU_MCMC_C(
  Observations = test_data$obs,
  Beta = rep(0, 10),
  Data_Matrix = t(test_data$design_mat),
  Iterations = 10000,
  Proposal_sd = 0.1)
  )



test_output <-pnorm(test_data$design_mat %*% test_data$beta)
  
