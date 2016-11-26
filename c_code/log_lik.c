// Compile with -lgslcblas, -lgsl, -lm
#include <math.h>
#include <gsl/gsl_sf_erf.h>
#include <cblas.h>
#include <stdlib.h>

// 	tmp = malloc(*num_data * sizeof(double))
// 	free(tmp)

void log_lik(int *restrict num_data, int *restrict num_param, const double *restrict design_matrix, const int *restrict obs, const double *restrict beta, double *restrict tmp, double *restrict res){
	/*
	Computes the log likelihood function of the probit model, given a design matrix, 
	parameter vector and observed values. Stores the value in res.

 	tmp: a double array of size *num_data that is used for local computations. In this way,
	     no extra memory needs to be allocated for every log_likelihood computation.
	*/


	// Compute design_matrix * beta
	cblas_dgemv(CblasColMajor, CblasNoTrans, *num_data, *num_param, 1.0, design_matrix, *num_data, beta, 1, 0.0, tmp, 1);

	// Compute the erf values and sum them
	*res = 0;
	for(int i = 0; i < *num_data; i++){
		if(obs[i]) *res += log( (1 + gsl_sf_erf( tmp[i] / sqrt(2) )) / 2 );
		else *res += log( (1 - gsl_sf_erf(tmp[i] / sqrt(2) )) / 2 );
		// TODO: vectorize? Save in tmp and add 4 at a time, or use OPENBlas? 
		//       Look at time profiling.
	};
}


void augmented_density(int *restrict M, int *restrict num_data, int *restrict num_param, const double *restrict design_matrix, const int *restrict obs, const double *restrict beta, double *restrict tmp, double *restrict res){
	/* 
	Adds the log of a normal prior on the beta-vector to the log likelihood function. Constants of the distribution 
	are ommitted since they cancel out in the Metropolis-Hastings algorithm.
	WARNING: if these are important, they need to be implemented!
	*/
	log_lik(num_data, num_param, design_matrix, obs, beta, tmp, res);
	*tmp = 0;
	size_t i = 0;
	for(; i < *num_param - 4; i += 4){
		*tmp +=  *(beta + i) * *(beta + i) + *(beta + i + 1) * *(beta + i + 1) 
		      +  *(beta + i + 2) * *(beta + i + 2) + *(beta + i + 3) * *(beta + i + 3);
	// TODO: Can we test if this is actually faster? I'm curious.
	}
	for(;i < *num_param; i++){
		*tmp += *(beta + i) * *(beta +i);
	}

	*res += - *tmp / 2 / *M;
}
	

