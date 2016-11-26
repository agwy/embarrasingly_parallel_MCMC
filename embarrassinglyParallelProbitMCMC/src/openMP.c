#include<omp.h>
#include<stdlib.h>

void openMP(const int *restrict num_iter, 
            const int *restrict num_data, const int *restrict num_par, 
            double *restrict design_matrix, const int *restrict obs,
            double *restrict proposal_sd, double *restrict init_value, 
            const int *restrict M, double *restrict tmp,
            double *acc_rate,
            double *restrict res){
  
  int num_per_subset = *num_iter / *M; //elements per subset

  //we need a pointer to the start of each design matrix subset;
  double *ptr_x[*M];
  double *ptr_obs[*M];
  
  for(size_t m=0; m < *M; m++){
    ptr_x[m] = &design_matrix[m*num_per_subset]; 
    ptr_obs[m] = &obs[m*num_per_subset];
  }
  
#pragma omp parallel for
  for(size_t m = 0; m  < *M; m++){
    MCMC(num_iter, num_data, num_par, ptr_x[m], ptr_obs[m],
         proposal_sd, init_value, M, tmp + m*(*num_data), acc_rate + m, res + (m*(*num_iter+1)));
  }

}