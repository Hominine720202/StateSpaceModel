data {
  int n;
  int n_comp1;
  int n_comp2;
  int comp_id1[n_comp1];
  int comp_id2[n_comp2];
  int miss_id1[n - n_comp1];
  int miss_id2[n - n_comp2];
  vector[n] x;
  vector[n_comp1] y_comp1;
  vector[n_comp2] y_comp2;
  real s_w2;
}

parameters {
  real beta;
  real mu_zero;
  real gamma_zero;
  real<lower=0> s_w1;
  //real<lower=0> s_w2;
  real<lower=0> s_v1;
  real<lower=0> s_v2;
  vector[n] mu;
  vector[n] gamma;
  vector[n - n_comp1] y_miss1;
  vector[n - n_comp2] y_miss2;
}

transformed parameters {
  vector[n] y1_mean;
  vector[n] y2_mean;
  vector[n] y1;
  vector[n] y2;
  
  // prepare y as the mixture of parameters and data
  y1[comp_id1] = y_comp1;
  y1[miss_id1] = y_miss1;
  y2[comp_id2] = y_comp2;
  y2[miss_id2] = y_miss2;
  
  y1_mean = mu + beta * x;
  y2_mean = gamma .* (mu + beta * x);
}

model {
  // use strong prior for gamma mean and sd
  gamma_zero ~ normal(1.5, 0.2);
  s_w2 ~ normal(0.02, 0.01);
  s_w1 ~ normal(0.1, 0.05);
  // state equation
  mu[1] ~ normal(mu_zero, s_w1);
  gamma[1] ~ normal(gamma_zero, s_w2);
  mu[2:n] ~ normal(mu[1:(n-1)], s_w1);
  gamma[2:n] ~ normal(gamma[1:(n-1)], s_w2);
  // output equation
  y1 ~ normal(y1_mean, s_v1);
  y2 ~ normal(y2_mean, s_v2);
}

