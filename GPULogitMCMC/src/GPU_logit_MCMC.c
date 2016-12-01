__global__ void LL_logit_GPU_1(int* obs, float* beta, float* data_mat, float* Log_like_GPU, int n, int p) {
  // @param p - the dimension of the parameter vector
  // @param n - the number of observations
  // @return An array with the log-likelihood of each observation given a parameter vector, for the logit model.
  
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(k < n){
    Log_like_GPU[k] = 0.0f;
    for(int j = 0;j < p;j++){
      // Note that the data_mat has been transposed:
      Log_like_GPU[k] +=  data_mat[k*p + j]*beta[j]; // replace by cuBlas?
    }
    
    if(obs[k] == 1){
      // log1pf(x) performs the single precision computation of log(1+x)
      Log_like_GPU[k] = -log1pf(expf(-Log_like_GPU[k]));    
    }else{
      Log_like_GPU[k] = -log1pf(expf(Log_like_GPU[k]));
    }
  }
}
 

extern "C" {
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include<gsl/gsl_sf_exp.h>
  
  void GPU_logit_MCMC(int* obs,
                float* beta, 
                float* data_mat,  
                float* Chain_output,
                float* Acceptance_Rate,
                int* n, 
                int* p, 
                int* Iter, 
                float* proposal_sd ) {
    // Now no explicit prior is being used. Change to normal prior?
    
    //Setup RNG
    static gsl_rng * r = NULL ;
    if(r == NULL) { // First call to this function, setup RNG
      gsl_rng_env_setup();
      r = gsl_rng_alloc(gsl_rng_mt19937);
    }
    
    //Pointers for GPU  
    float *beta_GPU, *data_GPU, *Log_like_GPU, *Log_like_vec, *Loglike, *Proposed_LL, *Proposal_beta;
    int *obs_GPU;
    int grid_size,block_size,Accepted = 0;
    
    Log_like_vec =  (float*) malloc(*n*sizeof(float));
    Proposal_beta = (float*) malloc((*p)*sizeof(float));
    Loglike = (float*) malloc(sizeof(float));
    Proposed_LL = (float*) malloc(sizeof(float));
    
    // Full first element of chain
    for(int j =0; j < *p ;j++){
      Chain_output[j] = beta[j];  
    }
    
    //Number of threads within each thread block
    block_size = 1024;
    
    //Number of thread blocks in each grid
    grid_size = (int) ceil( (float)*n / block_size );
    
    //Allocate storage on GPU
    //Perm
    cudaMalloc(&obs_GPU, *n*sizeof(int));
    cudaMalloc(&data_GPU, (*n)*(*p)*sizeof(float));
    
    cudaMemcpy(obs_GPU, obs, *n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(data_GPU, data_mat, (*n)*(*p)*sizeof(float), cudaMemcpyHostToDevice);
    
    //varied
    cudaMalloc(&beta_GPU, *p*sizeof(float));
    cudaMalloc(&Log_like_GPU, *n*sizeof(float));
    
    //Start with inital beta
    cudaMemcpy(beta_GPU, beta, *p*sizeof(float), cudaMemcpyHostToDevice);
    
    //Perform Computation
    LL_logit_GPU_1<<<grid_size, block_size>>>(obs_GPU, beta_GPU, data_GPU, Log_like_GPU, *n, *p);
    cudaThreadSynchronize();
    
    //Copy down Log_likelihood vector
    *Loglike = 0.0f;
    cudaMemcpy(Log_like_vec, Log_like_GPU, *n*sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int j =0; j < *n ;j++){
      *Loglike += Log_like_vec[j];
    }
    
    //Start MCMC iterations
    for(int j =1; j < *Iter ; j++){
      for(int k =0; k < *p; k++){
        Proposal_beta[k] =  Chain_output[ (j-1)*(*p) + k ] +  (float) gsl_ran_gaussian(r,*proposal_sd);
        
      }
      
      //Upload proposal to GPU
      cudaMemcpy(beta_GPU, Proposal_beta, *p*sizeof(float), cudaMemcpyHostToDevice);
      
      //Update calculate Log_like_GPU
      LL_logit_GPU_1<<<grid_size, block_size>>>(obs_GPU, beta_GPU, data_GPU,Log_like_GPU,*n,*p);
      cudaThreadSynchronize();
      
      //Copy down answer
      cudaMemcpy(Log_like_vec, Log_like_GPU, *n*sizeof(float), cudaMemcpyDeviceToHost);
      
      //bring together log likelihoods
      *Proposed_LL = 0.0f;
      for(int k =0; k< *n ;k++){
        *Proposed_LL += Log_like_vec[k];
      }
      
      
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
  }
}