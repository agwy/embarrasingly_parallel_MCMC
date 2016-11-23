setwd("/data/treecreeper/gouwy/OxWaSP/Module_4/Project")
library(parallel)

num_cores = 4

mclapply(list_of_targets, mcmc_algorithm, mc.cores = num_cores) # Needs a valid mcmc algorithm




