cd(@__DIR__)
using DifferentialEvolutionMCMC, Random, Distributions

Random.seed!(50514)

priors = (
    μ = (Normal(0, 10),),
    σ = (Truncated(Cauchy(0, 1), 0.0, Inf),)
)

bounds = ((-Inf,Inf),(0.0,Inf))

data = rand(Normal(0.0, 1.0), 50)

function loglike(data, μ, σ)
    return sum(logpdf.(Normal(μ, σ), data))
end

model = DEModel(; priors, model=loglike, data)

de = DE(;bounds, burnin=1000, priors, discard_burnin=false)
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
println(chains)
