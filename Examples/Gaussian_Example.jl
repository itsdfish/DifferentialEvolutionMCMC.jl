cd(@__DIR__)
using Revise, DifferentialEvolutionMCMC, Random, Distributions

Random.seed!(50514)

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

bounds = ((-Inf,Inf),(0.0,Inf))

data = rand(Normal(0.0, 1.0), 50)

function loglike(data, μ, σ)
    return sum(logpdf.(Normal(μ, σ), data))
end

names = (:μ,:σ)

model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)


de = DE(;bounds, burnin=1000, Np=6)
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)