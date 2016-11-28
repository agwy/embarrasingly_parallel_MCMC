// R CMD build embarrassinglyParallelProbitMCMC/
// R CMD INSTALL -l Packages/ embarrassinglyParallelProbitMCMC_0.1.0.tar.gz 
// C function for MH MCMC
// Compile with -lgslcblas, -lgsl, -lm
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_exp.h>
#include "module4.h"


void MCMC(const int *restrict num_iter, const int *restrict num_data, const int *restrict num_par, 
          const double *restrict design_matrix, const int *restrict obs,
          const double *restrict proposal_sd, const double *restrict init_value, 
	  const int *restrict M, double *restrict tmp,
          double *acc_rate,
          double *restrict res){
  
  static gsl_rng *restrict r = NULL;
  
  if(r == NULL) { //First call to this function, setup RNG
    gsl_rng_env_setup();
    r = gsl_rng_alloc(gsl_rng_mt19937);
  }

  *acc_rate = 0.0;
 
  
  //d is of length num_par
  //res is of dimesion num_iter x num_par
  
  double proposal[*num_par];
  double u = 0.0;
  double prop_dens = 0.0;
  double old_dens = 0.0;
  augmented_density(M, num_data, num_par, design_matrix, obs, init_value, tmp, &old_dens); 
    
  for(int i=0; i<*num_par; i++){
    res[i] = init_value[i]; /*by row*/
  }
    
 
  for(int i = 1; i <= *num_iter; i++){
    for(int d = 0; d < *num_par; d++){
        proposal[d] = res[(i-1)*(*num_par)+d] + gsl_ran_gaussian(r, *proposal_sd); 
    }
    
    u = gsl_rng_uniform(r);
    augmented_density(M, num_data, num_par, design_matrix, obs, proposal, tmp, &prop_dens); 
    
    if( prop_dens - old_dens > log(u) ){    
      old_dens = prop_dens; 
      for(int d=0; d<*num_par; d++){
        res[i*(*num_par)+d] = proposal[d];
      }
	*acc_rate += 1;
    }else{
      for(int d=0; d < *num_par; d++){res[i*(*num_par)+d] = res[(i-1)*(*num_par)+d];}
    }
  }
*acc_rate /= *num_iter;
    
}
