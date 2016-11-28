__global__ void LLGPU(int* obs, float* beta, float* data_mat,float* Log_like_GPU,int n,int p) {
  
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(k < n){
    Log_like_GPU[k] = 0.0f;
    for(int j = 0;j < p;j++){
      Log_like_GPU[k] +=  data_mat[k*p + j]*beta[j];
    }
    
    if(obs[k] == 1){
      Log_like_GPU[k] = logf(normcdff(Log_like_GPU[k]));    
    }else{
      Log_like_GPU[k] = logf(1 - normcdff(Log_like_GPU[k]));    
    }
  }
}

