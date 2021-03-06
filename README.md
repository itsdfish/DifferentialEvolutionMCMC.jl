[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5083368.svg)](https://doi.org/10.5281/zenodo.5083368)


# DifferentialEvolutionMCMC

DifferentialEvolutionMCMC.jl is a Differential Evolution MCMC sampler written in Julia and uses the AbstractMCMC interface. DifferentialEvolutionMCMC.jl works with any model, provided that it returns an exact or approximate log likeilhood. An annotated example is provided below. Other examples can be found in the examples subfolder.

## Example

First, load the required libraries.

```julia
using DifferentialEvolutionMCMC, Random, Distributions

Random.seed!(50514)
```

Define the prior distributions as a NamedTuple of distribution objects and number of elements, N: (distribution, N). Omit the number of elements if the parameter is a scalar.

```julia
priors = (
    μ = (Normal(0, 10),),
    σ = (Truncated(Cauchy(0, 1), 0.0, Inf),)
)
```

Specify the upper and lower bounds of the parameters.

```julia
bounds = ((-Inf,Inf),(0.0,Inf))
```

Generate simulated data from a normal distribution

```julia
data = rand(Normal(0.0, 1.0), 50)
```

Next, define a function for the log likelihood which accepts the data follow by the parameters (in the order specififed in the priors). Create a second method that accepts a vector of parameters and maps it to the original function and makes a reference to the data.

```julia
function loglike(data, μ, σ)
    return sum(logpdf.(Normal(μ, σ), data))
end
```

Note that in some cases it might be preferable to pass your parameters as a vector rather than specify each parameter in the function definition. In such cases, you can use `loglike(data, θ...)`.

Create a model object containing the prior and loglikelihood function and create a differential evolution object. Default settings can be overriden with keyword arguments.

```julia
model = DEModel(;priors, model=loglike, data)

de = DE(;bounds, burnin=1000, priors)

```

To run the sampler, pass the model and differential evolution object along with settings for the number iterations and MCMCMThreads() for multithreading.

```julia
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
println(chains)
```
