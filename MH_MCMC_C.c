// C function for MH MCMC

#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_sf_exp.h>


void MCMC(const int *restrict num_iter, const int *restrict num_data, const int *restrict num_par, 
          double *restrict design_matrix, const int *restrict obs, double *restrict beta,
          double *restrict proposal_sd, double *restrict init_value, double *acc_rate,
          double *restrict res){
  
  static gsl_rng *restrict r = NULL;
  
  if(r == NULL) { //First call to this function, setup RNG
    gsl_rng_env_setup();
    r = gsl_rng_alloc(gsl_rng_mt19937);
  }
  
  *acc_rate = 0;
  
  //d is of length num_par
  //res is of dimesion num_iter x num_par
  
  double proposal[*num_par];
  double prev_beta[*num_par];
  double rnorm[*num_par];
  
  for(int i=0; i<*num_par; i++)
  res[i] = init_value[i]; /*by row*/
  
  for(int i = 1; i < *num_iter; i++){
    for(int d = 0; d < *num_par; d++){
      
      rnorm[d] =  gsl_ran_gaussian (r, proposal_sd[d]); /*simulate rnorm[*num_par] vector eacht time;*/
      proposal[d] = res[(i-1)*(d+1)+d] + rnorm[d]; 
      prev_beta[d] = res[(i-1)*(d+1)+d];
    }
    double u = gsl_rng_uniform(r);
    if(gsl_sf_exp(target_density(*num_data, *num_par, *design_matrix, *obs, *proposal) /*supply beta_prop;*/
             - target_density(*num_data, *num_par, *design_matrix, *obs, *prev_beta)) /*prev_beta;*/
         > u){
      for(int d=0; d<*num_par; d++)
      res[i*(*num_par)+d] = proposal[d];
      
      *acc_rate += 1;
    } else {
      for(int d=0; d<*num_par; d++)
        res[i*(*num_par)+d] = res[(i-1)*(*num_par)+d];
    }
    *acc_rate /= *num_iter; 
  }
  
}
