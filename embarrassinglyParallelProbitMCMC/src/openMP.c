#include<omp.h>
#include<stdlib.h>
#include<stdio.h>
#include "module4.h"


void openMP(const int *restrict num_iter, 
            const int *restrict num_data, const int *restrict num_par, 
            double *restrict design_matrix, int *restrict obs,
            const double *restrict proposal_sd, const double *restrict init_value, 
            const int *restrict M, double *restrict tmp,
            double *acc_rate,
            double *restrict res){
  
  int num_per_subset = *num_data / *M; //elements per subset

  //we need a pointer to the start of each design matrix subset;
  double *ptr_x[*M];
  int *ptr_obs[*M];
  
  for(int m=0; m < *M; m++){
    ptr_x[m] = &design_matrix[m * (*num_par) * num_per_subset]; 
    ptr_obs[m] = &obs[m * num_per_subset]; 
  }
  
#pragma omp parallel for
  for(int m = 0; m  < *M; m++){
    //printf("pointer %p", ptr_x[m]);
    MCMC(num_iter, &num_per_subset, num_par, ptr_x[m], ptr_obs[m],
         proposal_sd, init_value, M, &tmp[m*(num_per_subset)], &acc_rate[m], &(res[(m * (*num_par) * (*num_iter+1))]));
  }
}
