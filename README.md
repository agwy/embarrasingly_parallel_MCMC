# embarrasingly_parallel_MCMC

Only 1 s: it illustrates it is embarrassing right...


## Research questions:

1. What is the effect multi-threading? I.e. running more threads than cores. Compare times and print something at the beginning and ending of execution of a thread.
2. Why is the R code still faster than the C code? Vectorization of the log-likelihood? OPENBlas?
3. Methodology comparisons: Compare subsetting vs. no subsetting of data; (compare kernel estimates to averaging? less keen on this...)


TODO: Solve c profiling!
