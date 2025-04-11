# Bayesian inference under sparsity for spatial binary regression models .

We implement a spatial binary regression model for area data, using a class of flexible link functions that include an extra shape parameter that can be adapted according to the degree of skewness present in the data. In the spatial part, we assign the **DAGAR model** (directed acyclic graph autoregressive) to the precision matrix $\Omega$ of spatial random effects. The provided codes were written in R and also in C++ to optimize computational efficiency through the R package `Rcpp`.

## Overview
The implementations performed here derive from the results obtained from the doctoral dissertation, currently under finalization, of Alan S. Assunção, under the supervision of Ricardo S. Ehlers. We illustrate our application to a motivating dataset on periodontal disease and also include a simulation study to investigate the robustness of our methods. We follow a Bayesian approach to perform parameter estimation and model comparison.

## A Hierarchical Spatial Model

Consider a spatial situation where we observe a binary response $y_{is}$ for subject $s$, at site $i$ within subject $s$. We assume that $Y_{si}\sim\mbox{Bernoulli}(p_{si})$ and for each individual $s$ the probability that $Y_{si}=1$ depends on a set of subject level covariates $x_s$ and on a neighbouring structure. The binary regression model with a spatial component is then given by

$$p_{si} = F_\lambda(x_{s}'\beta + \phi_{si}), i=1,\dots,m, ~s=1,\dots,r$$
$$\delta=log(\lambda) \sim U(-2,2) \mbox{ and } \beta \sim N_p(0,100\times I_p)$$
$$\phi_s \sim N_m(0,(I-B)^\top_\pi F_{\pi}(I-B)_{\pi}))\mbox{ } s=1,\ldots,r,$$

were precision matrix of spatial random effects $\phi_s$ (for $s=1,\ldots,r$), is the DAGAR model presented in [Datta et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8046356/); $I$ is identity matrix with dimensions $m\times m$, $B$ is a strictly lower triangular matrix,  $F=dia(\tau_1,\ldots,\tau_m)$ is a diagonal matrix (for more details see [Datta et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8046356/)) , and $F_\lambda$ denotes a continuous cumulative distribution function (cdf), which can be any monotonically increasing function that maps an input in $\mathbb{R}$ onto the (0,1) interval and $F^{-1}$ is typically called a link function. $\lambda$ is the shape parameter, $x_{s}$ is the vector of covariates for subject $i$ (that do not vary across space), $\beta\in\mathbb{R}^k$ is the $k\times 1$ vector of covariate coefficients (fixed effects) and $\phi_{si}$ are spatially correlated random effects.

$\delta=log(\lambda) \in \mathbb{R}$, is a convenient parameterization for the shape parameter $\lambda$ to simplify Bayesian computation, as per Bazán et al. (2017), $S$ is the scale matrix and $\kappa$ the degrees of freedom. We will be using the class of Power and Power Reverse link functions presented in the work of [Bazán et al. (2017)](https://onlinelibrary.wiley.com/doi/pdf/10.1002/asmb.2215) to deal with unbalanced data scenarios. The $\rho$ (spatial correlation) and $\tau_\phi$ (marginal precision) values ​​were set to $0.9$ and $1$, respectively.
## Main Functions

### Implementação do amostrador de Monte Carlo Hamiltoniano para $\beta$, $\delta$ e $\phi_i$ para $i=1,\ldots,m$

For each of the link functions adopted in this application, there are two files containing functions to enable the implementation of the models under the respective link function. The first file contains `R` functions that will calculate the information matrix of the model under the desired link function for the parameter vector $(\beta,\delta)$. The information matrix is ​​necessary for the implementation of the Riemann-Manifold Hamiltonian Monte Carlo, which will sample the parameter vector $(\beta,\delta)$. The second file, with the extension `.cpp`, contains functions that implement, strictly speaking, the Hamiltonian Monte Carlo methods to sample the parameter vector $(\beta,\delta)$ and the spatial random effects vectors $\phi_i$ $i=1,\ldots,n$. The description of these files for each link function is shown in the table below.

Link | R file | Rcpp file
---  |---     |---
Power Cauchy          | funcoes-cauchy-potencia-R.R         | hmcCpp-cauchy-potencia.cpp
Reverse Power Cauchy  | funcoes-cauchy-reversa-potencia-R.R | hmcCpp-cauchy-reversa-potencia.cpp
Power Logistic        | funcoes-logistica-potencia-R.R      | hmcCpp-logistica-potencia.cpp
Reverse Power Logistic| funcoes-logistica-reversa-potencia-R.R |hmcCpp-logistica-reversa-potencia.cpp
Power Reverse Gumbel  | funcoes-gumbel-reversa-potencia-R.R | hmcCpp-gumbel-reversa-potencia.cpp
Reverse Power Reverse Gumbel | funcoes-gumbel-reversa-reversa-de-potencia-R.R | hmcCpp-gumbel-reversa-reversa-de-potencia.cpp
Power Normal          | funcoes-normal-potencia-R.R         | hmcCpp-normal-potencia.cpp
Reverse Power Normal  | funcoes-normal-reversa-potencia-R.R | hmcCpp-normal-reversa-potencia.cpp
Logit                 | funcoes-logito-R.R                  | hmcCpp-logito.cpp
Probit                | funcoes-probito-potencia-R.R        | hmcCpp-probito-potencia.cpp
Cloglog               | funcoes-cloglog-potencia-R.R        | hmcCpp-cloglog-potencia.cpp

Each `R` file above is composed of four functions:
* `F` - implements the cumulative distribution function that gives rise to the respective link function
* `lpostbetadelta` - implements the log-posteriori of the parameter vector $(\beta,\delta)$
* `gradbetadelta` - implements the gradient of the parameter vector $(\beta,\delta)$ under the respective link function
* `G` - Calculates the information matrix of the model under the specified link function

Each file with `Rcpp` functions described in the table is composed of the first three functions described above, but "translated" to `Rcpp`, together with the following functions:
* `lpostphi` - implements the log-posteriori of the spatial random effects vector $\phi_i$ para $i=1,\ldots,n$
* `gradphi` - implements the gradient of the spatial random effects vector $\phi_i$ para $i=1,\ldots,n$
* `hmcCpp` - implements Hamiltonian Monte Carlo methods to sample the parameter vector $(\beta,\delta)$ and spatial random effects $\phi_i$ $i=1,\ldots,n$

### Sampling from G-Wishart

To sample values ​​from the G-Wishart distribution, we will use a function available in the package `R` `BDgraph` [Mohammadi, R., Massam, H., & Letac, G. (2021).](https://www.tandfonline.com/doi/full/10.1080/01621459.2021.1996377).

## Auxiliary functions
The file **funcoes-auxiliares.R** contains two functions:

* `W_sparsa` - Calculation of various quantities related to the adjacency matrix W. Return a list of results, namely: number of area units; number of adjacency pairs; adjacency pairs; and number of neighbors for each area unit
* `adjacency2` - A function that imports the adjacency matrix, formatting it and making it ready to be used. This function is designed to import the adjacency matrix when it has been saved in .csv format.

## Example
We consider a sample of $n = 100$ subjects, $m = 60$ sites for each subject, $\lambda=2$ and two covariates $x_1$ and $x_2$ generated from a distribution $N(0,1)$ each. The observed data $Y_{is}$ with $i=1,\ldots,100$ and $s=1,\ldots,60$ were generated using the link function of the Power Cauchy distribution 
$$p_{is}=\left[\frac{1}{\pi}\mbox{arctan}(x_i^\top\beta+\phi_{is})+\frac{1}{2}\right]^\lambda$$. True regression parameter : $\beta = (-0.7, 0.7)$. 5000 MCMC samples were generated, discarding 2500 of them as burn-in

The simulated data can be found here in the simulated-data folder

**Calling the data and loading the adjacency matrix**

```R
rm(list=ls())# clear PC memory

###################################################################################
# Required packages
###################################################################################

library('BDgraph')	# to sample from G-wishart
library ('Rcpp')
library ('microbenchmark')
library ('rbenchmark')
library ('RcppArmadillo')

# R functions 

source("funcoes-cauchy-potencia-R.R")
source("funcoes-auxiliares.R")

#  Rcpp/C++ functions

sourceCpp("hmcCpp-cauchy-potencia.cpp")

#################################################### Example of use #######################################

y=read.table("Y.txt",header=TRUE) # binary responses

Y=as.matrix(y)

x=read.table("X.csv",header=TRUE,sep=',') # covariates

X=as.matrix(x)

pcov = length(X[1,])

m = dim(Y)[2]

n = dim(Y)[1]

W=adjacency2('/home/alan/Documentos/TESE-ALAN/Artigo-regressao-binaria-espacial/ligacao-Cauchy-Potencia/W_.csv',m)

W_esparsa =W_sparsa(W) # calculating quantities for the sparse adjacency matrix W

D = diag(as.vector(W_esparsa$D_sparse)) # diagonal matrix with neighbors | random grid

rho = 0.9 # spatial correlation coefficient

S = D-rho*W #  CAR priori -
```

**Defining the quantities to run the Hamiltonian Monte Carlo and obtaining the maximum a posteriori of $(\beta,\delta)$**

```R
#################################### HMC SETTINGS AND PREPARING THE SIMULATION #############################

SS = 2500       # chain size at the end of simulations
burn = 2500      # burn-in
lagg =1         # lagg

SS. = (SS + burn)*lagg         # chain size
idx = seq(burn * lagg + 1, SS., by = lagg)
      
kap = m # kappa fixed for simulations
      
Omega_sim=rgwish( n = 1, adj =W, b = kap, D = S, threshold = 1e-8 ) # generating the initial precision matrix

phi=mvrnorm(n, mu = rep(0, times =m), Sigma = solve(Omega_sim)) # generating the initial spatial random effects

betainit = rep(1,3) # rep(1,5)

deltainit = -0.1

para = c(deltainit,betainit)
     
map = optim(para, lpostbetadelta, gradbetadelta,phi, y = Y, X=X, control = list(fnscale = -1), method = 'BFGS', hessian = TRUE); map
      
G. = G(theta=c(round(map$par,2),phi),y=Y,X=X) # Fisher information matrix of the model
      
theta.current =c(map$par,phi)
      
D. <- length(theta.current)
      
theta      <- matrix( , SS., D.)
theta[1, ] <- theta.current
```
**Simulating posteriori samples**
```R
      
######################################### INITIAL VALUES FOR PARAMETER CHAINS ######################
      
kgw = (m+n)
Sgw = S
      
Omegacpp = Omega_sim

rbetadelta = 0

################################################## SAMPLING PARAMETERS ##############################################################
tempo=system.time(
        for(r in 2:SS.) {
          
          # sampling beta, xi e phi i=1,...,n
          
          B=hmcCpp(theta[r-1,],SS=1,burn=1,lag=1, epsilon=0.1, LF=22,epsphi = 0.001, Lphi = 22, M=G., y=Y,X=X, Omega=Omegacpp)
          
          theta[r,] = B$theta
          
          #  rbetaphi = rbetaphi + 2*B$"taxa-aceitacao phi"
          
          rbetadelta = rbetadelta + 2*B$"taxa-aceitacao beta-delta"
          
          phimc = matrix(theta[r,(2+pcov):(1+pcov+n*m)],nrow=n,ncol=m)  # spatial effects
          
          for (q in 1:n)
          {            
            Sgw = Sgw + phimc[q,]%*%t(phimc[q,]) # log-priori de phi  (implementing it this way is equivalent to calculating the trace of
# matrix multiplication)
          }
          
          Sgw = S+Sgw
          
          # sampling Omega
          
          Omegacpp=rgwish( n = 1, adj =W, b = kgw, D = Sgw, threshold = 1e-8 ) # generating the precision matrix
          
          print(r)
          
          Sgw = 0
        }         
)
```
**Obtaining a posteriori estimates**

```R
 theta = theta[idx, ]
      
post = theta
      
deltapost = post[,1]
      
lambdapost = exp(deltapost)
      
betapost = post[,(2:(pcov+1))]
      
phipost = post[,((2+pcov):(1+pcov+n*m))]
      
# Posterior mean of beta and lambda

hat.lambda=mean(lambdapost)
hat.beta = colMeans(betapost)


# 95% credible interval of beta and lambda
CI.beta = apply(cbind(lambdapost,betapost), 2, quantile, probs=c(0.025,0.975))
```
**Graph showing true and estimated regression and shape parameters along with 0.95 credibility intervals:**

```R
# data frame for plotting the posterior estimates of beta

true.beta=c(2,-0.7,0.7)

df_beta = data.frame(x = paste0(c('lambda','beta 1','beta 2')), beta = c(true.beta, hat.lambda,hat.beta),
                     type = rep(c("true", "estimated"), each = pcov+1),
                     beta_u = CI.beta[2,], beta_l = CI.beta[1, ])

# plot the posterior mean and 95% CI of beta, along with the true parameters 
g.beta = ggplot(df_beta, aes(x = x, y = beta)) +
  geom_errorbar(aes(ymin=beta_l, ymax=beta_u), width=.15, color = "red3") +
  geom_point(aes(color = type, shape = type)) +
  scale_color_manual(values = c("red3", "black"))+
  geom_abline(slope = 0, linetype = "dotted")+
  xlab("") + ylab("")+theme_bw()
g.beta
```
![](https://github.com/user-attachments/assets/d017ec3c-ac9e-4dbe-8d73-a5ceada63c30)

**Chain convergence graphs**

```R

# trace-plot

par(mfrow=c(1,3))
plot(lambdapost,xlab = '',ylab = '',main = 'lambda', type = 'l');
plot(betapost[, 1],xlab = '',ylab = '',main = 'Beta1', type = 'l');
plot(betapost[, 2],xlab = '',ylab = '',main = 'Beta2', type = 'l')
```
![](https://github.com/user-attachments/assets/ea7251c7-5642-4539-a3ab-3cb2e8bc9c7c)

```R

# auto correlation plot

par(mfrow=c(1,3))
acf(lambdapost,xlab = '',ylab = '',main = 'lambda', type = "correlation") 

acf(betapost[,1],xlab = '',ylab = '',main = 'Beta1', type = "correlation") 

acf(betapost[,2],xlab = '',ylab = '',main = 'Beta2', type = "correlation") 

```
![](https://github.com/user-attachments/assets/ab83a8da-5914-4569-b44e-81d8acf28f26)


## Reference
* Alves, J. S., Bazán, J. L. and Arellano-Valle, R. B. (2023) Flexible cloglog links for binomial regression models as
an alternative for imbalanced medical data. Biometrical Journal, 65, 2100325.
* Bazán, J., Torres-Avilés, F., Suzuki, A. K. and Louzada, F. (2017) Power and reversal power links for binary
regressions: An application for motor insurance policyholders. Applied Stochastic Models in Business and
Industry, 33, 22–34
* Mohammadi, A. and Wit, E. C. (2015) Bayesian structure learning in sparse gaussian graphical models.
* Girolami, M. and Calderhead, B. (2011) Riemann manifold Langevin and Hamiltonian Monte Carlo methods.
Journal of the Royal Statistical Society B, 73, 123–214.
