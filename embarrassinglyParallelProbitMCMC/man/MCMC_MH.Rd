% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/C_Wrapper_MCMC.R
\name{MCMC_MH}
\alias{MCMC_MH}
\title{MCMC_MH}
\usage{
MCMC_MH(M, Iterations, Data_Matrix, Obs, Inital_beta, proposal_sd)
}
\arguments{
\item{M}{- Number of subsets of the data.}

\item{Iterations}{- Number of iterations for the Markov chain.}

\item{Data_Matrix}{- A N x P matrix. The design matrix of the probit model, i.e. usually the X-matrix.}

\item{Obs}{- A vector of length N. The observations of the logit model.}

\item{proposal_sd}{- A number for the standard deviation used for the proposal. It is the same for all parameters.}

\item{Initial_beta}{- A vector of starting values for the chains, usually taken as a vector of zeroes.}
}
\value{
A list of [[1]] a matrix containing all the steps of the MH chain of dimension (Iterations+1) x d,
                         where d is the number of parameters, i.e. length of the vector initial_value,
                         and the first row is the Initial_beta;
                 [[2]] the acceptance rate for the chain.
}
\description{
MCMC_MH
}

