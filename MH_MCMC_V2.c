
// C function for MH MCMC

#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_sf_exp.h>


void MCMC(const int *restrict num_iter, const int *restrict num_data, const int *restrict num_par, 
          double *restrict design_matrix, const int *restrict obs,
          double *restrict proposal_sd, double *restrict init_value, 
          double *acc_rate,
          double *restrict res){
  
  static gsl_rng *restrict r = NULL;
  
  if(r == NULL) { //First call to this function, setup RNG
    gsl_rng_env_setup();
    r = gsl_rng_alloc(gsl_rng_mt19937);
  }
  
  for(int i =0;i<*num_iter;i++){
    acc_rate[i] = 0.0;  
  }
  
  
  //d is of length num_par
  //res is of dimesion num_iter x num_par
  
  double proposal[*num_par];
  double prev_beta[*num_par];
  double u = 0.0;
    
  for(int i=0; i<*num_par; i++){
    res[i] = init_value[i]; /*by row*/
  }
    
  
    for(int i = 1; i <= *num_iter; i++){
      for(int d = 0; d < *num_par; d++){
        
          proposal[d] = res[(i-1)*(d+1)+d] + gsl_ran_gaussian (r, *proposal_sd); 
          prev_beta[d] = res[(i-1)*(d+1)+d];
      }
      u = gsl_rng_uniform(r);
      if(0.2 > u){
        /*gsl_sf_exp(target_density(*num_data, *num_par, *design_matrix, *obs, *proposal) 
                     - target_density(*num_data, *num_par, *design_matrix, *obs, *prev_beta))*/
        
        for(int d=0; d<*num_par; d++){
          res[i*(*num_par)+d] = proposal[d];
        }
        
        acc_rate[i-1] = 1;
        
      }else{
        
        for(int d=0; d < *num_par; d++){res[i*(*num_par)+d] = res[(i-1)*(*num_par)+d];}
      }
    }
    
}