[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5083368.svg)](https://doi.org/10.5281/zenodo.5083368)


# DifferentialEvolutionMCMC

DifferentialEvolutionMCMC.jl is a Differential Evolution MCMC sampler written in Julia and uses the AbstractMCMC interface. DifferentialEvolutionMCMC.jl works with any model, provided that it returns an exact or approximate log likeilhood. An annotated example is provided below. Other examples can be found in the examples subfolder.

## Example

First, load the required libraries.

```julia
using DifferentialEvolutionMCMC, Random, Distributions

Random.seed!(50514)
```

Define a function that returns the prior log likelihood of parameters μ and σ. Note
that order matters for parameters throughout. The algorithm expects parameters to have
the same order.

```julia
function prior_loglike(μ, σ)
    LL = 0.0
    LL += logpdf(Normal(0, 1), μ)
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
    return LL
end
```

Define a function for the initial sample. Sampling from the prior distribution is
a reasonable choice for most applications.

```julia
function sample_prior()
    μ = rand(Normal(0, 1))
    σ = rand(truncated(Cauchy(0, 1), 0, Inf))
    return [μ,σ]
end
```

Next, define a function for the log likelihood which accepts the data follow by the parameters (in the order specififed in the priors).

```julia
function loglike(data, μ, σ)
    return sum(logpdf.(Normal(μ, σ), data))
end
```

Specify the upper and lower bounds of the parameters.

```julia
bounds = ((-Inf,Inf),(0.0,Inf))
```
Define the names of parameters. Elements of parameter vectors do not need to be named.

```julia
names = (:μ,:σ)
```

Generate simulated data from a normal distribution

```julia
data = rand(Normal(0.0, 1.0), 50)
```

Now we will create a model object containing the sampling and log likelihood functions, the data and parameter names.

```julia
model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)

```

Next, define the DifferentialEvolution sampling object. This requires `bounds`, `burnin` and `Np`, which is the number of particles. 
```julia
de = DE(;bounds, burnin = 1000, Np = 6)
```
To run the sampler, pass the model and differential evolution object along with settings for the number iterations and MCMCMThreads() for multithreading.

```julia
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
println(chains)
```
