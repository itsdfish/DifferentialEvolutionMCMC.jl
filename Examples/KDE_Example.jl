cd(@__DIR__)
using DifferentialEvolutionMCMC, Random, KernelDensity, Distributions
include("KDE.jl")

function loglike(data, μ, σ)
    simdata = rand(Normal(μ, σ), 10_000)
    kd = kernel(simdata)
    dist = InterpKDE(kd)
    like = max.(1e-10, pdf(dist, data))
    return sum(log.(like))
end

# returns prior log likelihood
function prior_loglike(μ, σ)
    LL = 0.0
    LL += logpdf(Normal(0, 1), μ)
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
    return LL
end

# function for initial values
function sample_prior()
    μ = rand(Normal(0, 1))
    σ = rand(truncated(Cauchy(0, 1), 0, Inf))
    return [μ,σ]
end

# parameter names
names = (:μ,:σ)
# parameter bounds
bounds = ((-Inf,Inf),(0.0,Inf))

data = rand(Normal(0.0, 1.0), 50)


# model object
model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)

# DEMCMC sampler object
de = DE(;sample_prior, bounds, burnin = 1000, Np = 6)
# number of interations per particle
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)