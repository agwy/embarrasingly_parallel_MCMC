void log_lik(const int *restrict num_data, const int *restrict num_param, const double *restrict design_matrix, const int *restrict obs, const double *restrict beta, double *restrict tmp, double *restrict res);
void augmented_density(const int *restrict M, const int *restrict num_data, const int *restrict num_param, const double *restrict design_matrix, const int *restrict obs, const double *restrict beta, double *restrict tmp, double *restrict res);
void MCMC(const int *restrict num_iter, const int *restrict num_data, const int *restrict num_par, 
          const double *restrict design_matrix, const int *restrict obs,
          const double *restrict proposal_sd, const double *restrict init_value, 
          const int *restrict M, double *restrict tmp,
          double *acc_rate,
          double *restrict res)
  
void openMP(const int *restrict num_iter, 
              const int *restrict num_data, const int *restrict num_par, 
              double *restrict design_matrix, const int *restrict obs,
              double *restrict proposal_sd, double *restrict init_value, 
              const int *restrict M, double *restrict tmp,
              double *acc_rate,
              double *restrict res)