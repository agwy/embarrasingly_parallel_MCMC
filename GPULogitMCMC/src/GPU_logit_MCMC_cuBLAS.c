#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_exp.h>
#include <cublas_v2.h>

__global__ void logit_GPU(int* obs, float* Log_like_GPU, int n) {
  // @param p - the dimension of the parameter vector
  // @param n - the number of observations
  // @return An array with the log-likelihood of each observation given a parameter vector, for the logit model.
  
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(k < n){
    if(obs[k] == 1){
      // log1pf(x) performs the single precision computation of log(1+x)
      Log_like_GPU[k] = -log1pf(expf(-Log_like_GPU[k]));  
    }else{
      Log_like_GPU[k] = -log1pf(expf(Log_like_GPU[k]));
    }
  }
}

void GPU_logit_GPU_cuBLAS(cublasHandle_t handle, int *obs_GPU, float * beta_GPU, float * data_GPU, 
                          float * Log_like_GPU, float * Log_like_vec, float *Loglike, int n, int p, 
                          int grid_size, int block_size, const float * af, const float * bf) {
  // Store 'data_matrix %*% beta_vector' in Log_like_GPU:
  // cublasStatus_t stat; for debugging, set stat = cublasSgemv
  cublasSgemv(handle, CUBLAS_OP_N, n, p, af, data_GPU, n, beta_GPU, 1, bf, Log_like_GPU, 1);
  // if (stat != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed\n"); }
  cudaThreadSynchronize(); // Should this be here? Check if matrix multiplication is synchronous.

  // Compute the logistic function:
  logit_GPU<<<grid_size, block_size>>>(obs_GPU, Log_like_GPU, n);
  cudaThreadSynchronize();
  
  //Copy down Log_likelihood vector
  cudaMemcpy(Log_like_vec, Log_like_GPU, n*sizeof(float), cudaMemcpyDeviceToHost);
  
  *Loglike = 0.0f;
  for(int j =0; j < n ;j++){
    *Loglike += Log_like_vec[j];
  }
}


extern "C" {
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_exp.h>
#include <cublas_v2.h>
  
  void GPU_logit_MCMC_cuBLAS(int* obs,
                float* beta, 
                float* data_mat,  
                float* Chain_output,
                float* Acceptance_Rate,
                int* n, 
                int* p, 
                int* Iter, 
                float* proposal_sd ) {

    //Setup RNG
    static gsl_rng * r = NULL ;
    if(r == NULL) { // First call to this function, setup RNG
      gsl_rng_env_setup();
      r = gsl_rng_alloc(gsl_rng_mt19937);
    }
    
    //Pointers for GPU  
    float *beta_GPU, *data_GPU, *Log_like_GPU, *Log_like_vec, *Loglike, *Proposed_LL, *Proposal_beta;
    float af = 1.0f, bf = 0.0f; // Parameters for the cuBLAS multiplication.
    int *obs_GPU;
    int grid_size,block_size,Accepted = 0;
    
    Log_like_vec =  (float*) malloc(*n*sizeof(float));
    Proposal_beta = (float*) malloc((*p)*sizeof(float));
    Loglike = (float*) malloc(sizeof(float));
    Proposed_LL = (float*) malloc(sizeof(float));
    
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Full first element of chain
    for(int j =0; j < *p ;j++){
      Chain_output[j] = beta[j];  
    }

    //Number of threads within each thread block
    block_size = 1024;
    
    //Number of thread blocks in each grid
    grid_size = (int) ceil( (float)*n / block_size );
    
    //Allocate storage on GPU
    cudaMalloc(&obs_GPU, *n*sizeof(int));
    cudaMalloc(&data_GPU, (*n)*(*p)*sizeof(float));
    
    cudaMemcpy(obs_GPU, obs, *n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data_GPU, data_mat, (*n)*(*p)*sizeof(float), cudaMemcpyHostToDevice);
    
    //varied
    cudaMalloc(&beta_GPU, *p*sizeof(float));
    cudaMalloc(&Log_like_GPU, *n*sizeof(float));
    
    //Start with inital beta
    cudaMemcpy(beta_GPU, beta, *p*sizeof(float), cudaMemcpyHostToDevice);
    
    //Perform Computation; no transpose of data matrix!!
    GPU_logit_GPU_cuBLAS(handle, obs_GPU, beta_GPU, data_GPU, Log_like_GPU, Log_like_vec, Loglike, *n, *p, grid_size, block_size, &af, &bf);

    //Start MCMC iterations
    for(int j =1; j < *Iter ; j++){
      for(int k =0; k < *p; k++){
        Proposal_beta[k] =  Chain_output[ (j-1)*(*p) + k ] +  (float) gsl_ran_gaussian(r,*proposal_sd);
      }
      //Upload proposal to GPU
      cudaMemcpy(beta_GPU, Proposal_beta, *p*sizeof(float), cudaMemcpyHostToDevice);
      
      GPU_logit_GPU_cuBLAS(handle, obs_GPU, beta_GPU, data_GPU, Log_like_GPU, Log_like_vec, Proposed_LL, *n, *p, grid_size, block_size, &af, &bf);

      if( exp(*Proposed_LL - *Loglike) > (float) gsl_rng_uniform(r)  ){
        //Accept proposal 
        Accepted++;
        *Loglike = *Proposed_LL;
        for(int k =0; k < *p; k++){
          Chain_output[ (j)*(*p) + k ] = Proposal_beta[k];
        }
        
      }else{
        //reject proposal
        for(int k =0; k < *p; k++){
          Chain_output[ (j)*(*p) + k ] = Chain_output[ (j-1)*(*p) + k ];
        }
        
      }
      
    }
    
    *Acceptance_Rate = ((float) Accepted) / ((float) *n);
    
    //Free GPU memory
    cudaFree(obs_GPU);
    cudaFree(beta_GPU);
    cudaFree(data_GPU);
    cudaFree(Log_like_GPU);
    
    free(Log_like_vec);
    free(Proposal_beta);
    free(Loglike);
    free(Proposed_LL);
    
    // Destroy the handle
    cublasDestroy(handle);
  }
  

}