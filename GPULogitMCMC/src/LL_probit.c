
__global__ void LLGPU_1(int* obs, float* beta, float* data_mat, float* Log_like_GPU, int n, int p) {
  // @param p - the dimension of the parameter vector
  // @param n - the number of observations
  
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(k < n){
    Log_like_GPU[k] = 0.0f;
    for(int j = 0;j < p;j++){
      Log_like_GPU[k] +=  data_mat[k*p + j]*beta[j]; // replace by cuBlas?
    }
    
    if(obs[k] == 1){
      Log_like_GPU[k] = logf(normcdff(Log_like_GPU[k]));  
    }else{
      Log_like_GPU[k] = logf(1 - normcdff(Log_like_GPU[k])); 
    }
  }
}

extern "C" {
  #include <stdio.h>
  
  void GPU_Probit_LL(int* obs, float* beta, float* data_mat, float* Log_like , int* n, int* p) {
    
    float *beta_GPU, *data_GPU, *Log_like_GPU, *Log_like_vec;
    int* obs_GPU;
      
    int grid_size, block_size;
    
    Log_like_vec =  (float*) malloc(*n*sizeof(float)); // Remove to outside: no reallocation every iteration
    
    //Number of threads within each thread block
    block_size = 1024;
    
    //Number of thread blocks in each grid
    grid_size = (int) ceil( (float)*n/block_size );
    
    //Allocate storage on GPU
    cudaMalloc(&obs_GPU, *n*sizeof(int));
    cudaMalloc(&beta_GPU, *p*sizeof(float));
    cudaMalloc(&data_GPU, (*n)*(*p)*sizeof(float));
    cudaMalloc(&Log_like_GPU, *n*sizeof(float));
    
    //Copy data onto GPU
    cudaMemcpy(obs_GPU, obs, *n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(beta_GPU, beta, *p*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(data_GPU, data_mat, (*n)*(*p)*sizeof(float), cudaMemcpyHostToDevice);
    
    //Perform Computation
    LLGPU_1<<<grid_size, block_size>>>(obs_GPU, beta_GPU, data_GPU, Log_like_GPU, *n, *p);
    cudaThreadSynchronize();

    //Copy answer across
    *Log_like = 0.0f;
    cudaMemcpy(Log_like_vec, Log_like_GPU, *n*sizeof(float), cudaMemcpyDeviceToHost);
    
    
    for(int j =0; j< *n ;j++){
      *Log_like += Log_like_vec[j]; // vectorize?
    }
    
    //Free GPU memory
    cudaFree(obs_GPU);
    cudaFree(beta_GPU);
    cudaFree(data_GPU);
    cudaFree(Log_like_GPU);
    
    free(Log_like_vec);
    
  }
}
