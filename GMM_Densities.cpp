#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector GMM_LL_CPP(NumericVector x, NumericVector mu,NumericVector stand_dev,NumericVector probs,bool to_log) {
  int mixture_components = mu.size(),dat_points = x.size();
  NumericVector output(dat_points); //allocate storage for output
  double temp_val=0;
  
  for (int i = 0; i < dat_points; i++){
    temp_val=0;
    for (int j = 0; j < mixture_components; j++){
      temp_val = temp_val + probs[j]*R::dnorm(x[i],mu[j],stand_dev[j],0);
    }
    output[i] = temp_val;
  }
  if(to_log){
    return(log(output));
  }else{
    return(output);
  }
  
}
