```@setup gaussian_setup
using DifferentialEvolutionMCMC
using Random
using Distributions
using StatsPlots
Random.seed!(6541)

data = rand(Normal(0.0, 1.0), 50)

function prior_loglike(μ, σ)
    LL = 0.0
    LL += logpdf(Normal(0, 1), μ)
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
    return LL
end

function sample_prior()
    μ = rand(Normal(0, 1))
    σ = rand(truncated(Cauchy(0, 1), 0, Inf))
    return [μ,σ]
end

names = (:μ,:σ)
bounds = ((-Inf,Inf),(0.0,Inf))

function loglike(data, μ, σ)
    return sum(logpdf.(Normal(μ, σ), data))
end

model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)

de = DE(;sample_prior, bounds, burnin = 1000, Np = 6)

n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
```
# Gaussian Example

This simple example using a Gaussian model will guide you through the process of performing Bayesin parameter estimation with Differential Evolution MCMC. Suppose we observe $\mathbf{y} = \left[y_1,y_2, \dots, y_{50} \right]$ which are assumed to follow a Gaussian Distribution. Thus, we can write the following sampling statement:

$\mathbf{y} \sim \mathrm{normal}(\mu, \sigma)$

where $\theta$ is a parameter representing the probability of success. Our goal is to estimate the probability of success $\theta$ from the $k$. Let's assume the prior distribution of $theta$ is given by

$\mu \sim \mathrm{normal}(0, 1)$

In this simple case, the posterior distribution of $\theta$ has a simple closed-form solution:

$\sigma \sim \mathrm{Cauchy}(0, 1)_{0}^{\infty}$

## Load Packages
Our first step is to load the required packages including `StatsPlots.jl` for plotting the MCMC chain.
```@example gaussian_setup
using DifferentialEvolutionMCMC
using Random
using Distributions
using StatsPlots
Random.seed!(6541)
```
## Generate Data
Let's assume we make $N=50$ using $\mu = 0$ and $\sigma=1$: 
```@example gaussian_setup
data = rand(Normal(0.0, 1.0), 50)
```
## Define Prior Log Likelihood Function
The next step is to define a function that passes our parameters $\mu$ and $\sigma$ and evaluates
the log likelihood. The function to compute the prior log likelihood is as follows:
```@example gaussian_setup
function prior_loglike(μ, σ)
    LL = 0.0
    LL += logpdf(Normal(0, 1), μ)
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
    return LL
end
```
## Define Sample Prior Distribution
Each particle in DifferentialEvolution must be seeded with an initial value. To do so, we define a function that returns the initial value. A common approach is to use the prior distribution as it is intended to encode likely values of $\mu$ and $\sigma$. The function for sampling from the prior distribution of $\mu$ and $\sigma$ is given in the following code block:
```@example gaussian_setup
function sample_prior()
    μ = rand(Normal(0, 1))
    σ = rand(truncated(Cauchy(0, 1), 0, Inf))
    return [μ,σ]
end
```

## Define Log Likelihood Function
Next, we define a function to compute the log likelihood of the data. The first argument must be the data followed by a separate argument for each parameter, maintaining the same order specified in `prior_loglike`. In our case, we can write the log likelihood function with `data` followed by $\mu$ and $\sigma$:

```@example gaussian_setup
function loglike(data, μ, σ)
    return sum(logpdf.(Normal(μ, σ), data))
end
```
## Define Bounds and Parameter Names
We must define the lower and upper bounds of each parameter. The bounds are a `Tuple` of tuples, where the $i^{\mathrm{th}}$ inner tuple contains the lower and upper bounds of the $i^{\mathrm{th}}$ parameter. Note that the parameters must be in the same order as specified in `prior_loglike` and `loglike`. In the present case, we only have $\mu$ followed by $\sigma$:
```@example gaussian_setup
bounds = ((-Inf,Inf),(0.0,Inf))
```
We can also pass a `Tuple` of names for each parameter, again in the same order as specified in `prior_loglike` and `loglike`:

```@example gaussian_setup
names = (:μ,:σ)
```
## Define Model Object
Now that we have defined the necessary components of our model, we will organize them into a `DEModel` object as follows:
```@example gaussian_setup
model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names)
```
## Define Sampler
Next, we will create a sampler with the constructor `DE`. Here, we will pass the `sample_prior` function and the variable `bounds`, which constains the lower and upper bounds of each parameter. In addition, we will specify $1,000$ burnin samples, `Np=6` particles per group (default 4 particles per group).
```@example gaussian_setup
de = DE(;sample_prior, bounds, burnin=1000, Np=6)
```

## Estimate Parameter
The code block below runs the sampler for $2000$ iterations with each group of particles running on a separate thread. The progress bar is also set to display. 
```@example gaussian_setup
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
```

## Convergence

We will evaluate convergence using two methods. First, we will verify that $\hat{r} \leq 1.05$ in the chain summary above. Indeed, it is. Its also important to examine the trace plot for any evidence of non-stationarity, or getting stuck at the same value for long periods of time. In the left panel of the plot below, the samples for each chain display the characteristic behavior indicative of efficient sampling and convergence: they look like white noise, or a "hairy caterpillar". The plot on the right shows the posterior distribution of $\mu$ and $\sigma$ for each of the 12 chains.

```@example gaussian_setup
plot(chains, grid=false)
```