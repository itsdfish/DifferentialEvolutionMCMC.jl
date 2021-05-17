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

priors = (
    μ = (Normal(0, 10),),
    σ = (Truncated(Cauchy(0, 1), 0.0, Inf),)
)

bounds = ((-Inf,Inf),(0.0,Inf))

data = rand(Normal(0.0, 1.0), 50)

model = DEModel(;priors, model=loglike, data)

de = DE(;bounds, burnin=1000, priors)
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)