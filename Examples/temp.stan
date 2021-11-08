data { 
    // total number of data points per person
    int<lower=1> n_data;
    // number of subjects
    int<lower=1> n_subj;
    // number observations in total
    int<lower=1> n;
    // person index 
    int<lower=1> idx[n]; 
    vector[n] y;
    vector[n] x;
} 
parameters {
  real<lower=0> sigma;
  real<lower=0> sigma_beta0;
  real mu_beta0;
  //real mu_beta1;
  vector[n_subj] beta0;
}

model {
   mu_beta0 ~ normal(1, 1);
   //mu_beta1 ~ normal(.5, 1);
   sigma_beta0 ~ cauchy(0, 1);
   sigma ~ cauchy(0, 1);
   beta0 ~ normal(mu_beta0, sigma_beta0);

   for(i in 1:n){
       real mu = beta0[idx[i]];// + mu_beta1 * x[i];
       y[i] ~ normal(mu, sigma);
   }
}

generated quantities {
    vector[n] mu = beta0[idx]; //+ mu_beta1 * x;
    real pred = sd(y - mu);
}