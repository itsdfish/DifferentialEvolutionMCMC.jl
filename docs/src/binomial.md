```@setup binomial_setup
using DifferentialEvolutionMCMC
using Random
using Distributions
using StatsPlots
Random.seed!(88484)

data = (N = 10,k = 6)

prior_loglike(θ) = logpdf(Beta(1, 1), θ)

sample_prior() = rand(Beta(1, 1))

bounds = ((0,1),)
names = (:θ,)

function loglike(data, θ)
    (;N,k) = data
    return logpdf(Binomial(N, θ), k)
end

model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)

de = DE(;sample_prior, bounds, burnin=1000, Np=3, σ=.01)
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
```
# Binomial Example

This simple example using a binomial model will guide you through the process of performing Bayesin parameter estimation with Differential Evolution MCMC. Suppose we observe $N$ samples from a random binomial process and observe $k$ successes. Formally, this can be stated as:

$k \sim \mathrm{Binomial}(N, \theta),$

where $\theta$ is a parameter representing the probability of success. Our goal is to estimate the probability of success $\theta$ from the $k$. Let's assume the prior distribution of $theta$ is given by:

$\theta \sim \mathrm{beta}(1, 1).$

In this simple case, the posterior distribution of $\theta$ has a simple closed-form solution:

$\theta_{|k} \sim \mathrm{beta}(1 + k, 1 + N - k).$

We will use this fact to verify that the MCMC sampler is working correctly. 

## Load Packages
Our first step is to load the required packages including `StatsPlots.jl` for plotting the MCMC chain.
```@example binomial_setup
using DifferentialEvolutionMCMC
using Random
using Distributions
using StatsPlots
Random.seed!(88484)
```
## Define Data
Let's assume we make $N=10$ observations which produce $k=6$ successes. 
```@example binomial_setup
data = (N = 10,k = 6)
```
The true posterior distribution is given by:
```@example binomial_setup
true_posterior = Beta(1 + 6, 1 + 4)
```
The maximum likelihood estimate of $\theta$ is

$\hat{\theta}_\mathrm{MLE} = \frac{k}{N} = 0.60$

The mean of the posterior of $\theta$ will be slightly less because our prior distribution encodes one failure and one success:

```@example binomial_setup
mean(true_posterior)
```

## Define Prior Log Likelihood Function
The next step is to define a function that passes our parameter $\theta$ and evaluates
the log likelihood. The function to compute the prior log likelihood is as follows:
```@example binomial_setup
prior_loglike(θ) = logpdf(Beta(1, 1), θ)
```
## Define Sample Prior Distribution
Each particle in DifferentialEvolution must be seeded with an initial value. To do so, we define a function that returns the initial value. A common approach is to use the prior distribution as it is intended to encode likely values of $\theta$. The function for sampling from the prior distribution of $\theta$ is given in the following code block:
```@example binomial_setup
sample_prior() = rand(Beta(1, 1))
```

## Define Log Likelihood Function
Next, we define a function to compute the log likelihood of the data. The first argument must be the data followed by a separate argument for each parameter, maintaining the same order specified in `prior_loglike`. In our case, we can write the log likelihood function as follows:

```@example binomial_setup
function loglike(data, θ)
    (;N,k) = data
    return logpdf(Binomial(N, θ), k)
end
```
## Define Bounds and Parameter Names
We must define the lower and upper bounds of each parameter. The bounds are a `Tuple` of tuples, where the $i^{\mathrm{th}}$ inner tuple contains the lower and upper bounds of the $i^{\mathrm{th}}$ parameter. Note that the parameters must be in the same order as specified in `prior_loglike` and `loglike`. In the present case, we only have a single parameter $\theta$:
```@example binomial_setup
bounds = ((0,1),)
```
We can also pass a `Tuple` of names for each parameter, again in the same order as specified in `prior_loglike` and `loglike`:

```@example binomial_setup
names = (:θ,)
```
## Define Model Object
Now that we have defined the necessary components of our model, we will organize them into a `DEModel` object as follows:
```@example binomial_setup
model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names)
```
## Define Sampler
Next, we will create a sampler with the constructor `DE`. Here, we will pass the `sample_prior` function and the variable `bounds`, which constains the lower and upper bounds of each parameter. In addition, we will specify $1,000$ burnin samples, `Np=3` particles per group (default 4 particles per group) and proposal noise of `σ=.01`.
```@example binomial_setup
de = DE(;sample_prior, bounds, burnin=1000, Np=3, σ=.01)
```

## Estimate Parameter
The code block below runs the sampler for $2000$ iterations with each group of particles running on a separate thread. The progress bar is also set to display. 
```@example binomial_setup
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
```

## Evaluation
### Convergence

We will evaluate convergence using two methods. First, we will verify that $\hat{r} \leq 1.05$ in the chain summary above. Indeed, it is. Its also important to examine the trace plot for any evidence of non-stationarity, or getting stuck at the same value for long periods of time. In the left panel of the plot below, the samples for each chain display the characteristic behavior indicative of efficient sampling and convergence: they look like white noise, or a "hairy caterpillar". The plot on the right shows the posterior distribution of $\theta$ for each of the 12 chains.

```@example binomial_setup
plot(chains, grid=false)
```

### Accuracy 
Let's also see if the results are similar to the closed-form solution. Below, we see that the mean and standard deviation are similar to the chain summary in the `Estimate Parameter` section, suggesting that the MCMC sampler is working as expected.
```@example binomial_setup
mean(true_posterior)
```

```@example binomial_setup
std(true_posterior)
```