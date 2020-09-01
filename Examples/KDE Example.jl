cd(@__DIR__)
using DifferentialEvolutionMCMC, Random, KernelDensity, Distributions
include("KDE.jl")

loglike(μ, σ, data) = sum(logpdf.(Normal(μ, σ,), data))

priors = (
    μ = (Normal(0, 10),),
    σ = (Truncated(Cauchy(0, 1), 0.0, Inf),)
)

bounds = ((-Inf,Inf),(0.0,Inf))

data = rand(Normal(0.0, 1.0), 50)

model = DEModel(priors=priors, model=x -> loglike(x..., data))

de = DE(bounds=bounds, burnin=1000, priors=priors, Np=6, n_groups=1,
    generate_proposal=fixed)
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
display(chains)
