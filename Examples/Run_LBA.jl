cd(@__DIR__)
using Revise, DifferentialEvolutionMCMC, Random, Parameters
using SequentialSamplingModels, Distributions
Random.seed!(88484)

dist = LBA(ν = [3.0, 2.0], A = 0.8, k = 0.2, τ = 0.3)
choice, rt = rand(dist, 100)
min_rt = minimum(rt)

function _prior_loglike(ν, A, k, τ, min_rt)
    LL = 0.0
    LL += sum(logpdf(Normal(1, 5), ν))
    LL += logpdf(Normal(0.8, 0.2), A)
    LL += logpdf(Normal(0.2, 0.1), k)
    LL += logpdf(Uniform(0, min_rt), τ)
    return LL
end

function sample_prior()
    ν = rand(Normal(1, 5), 2)
    A = rand(Normal(0.8, 0.2))
    k = rand(Normal(0.2, 0.1))
    τ = rand(Uniform(0, min_rt))
    return as_union([ν, A, k, τ])
end

wrapper(min_rt) = (ν, A, k, τ) -> _prior_loglike(ν, A, k, τ, min_rt)

prior_loglike = wrapper(min_rt)

bounds = ((0.0, Inf), (0.0, Inf), (0.0, Inf), (0.0, min_rt))
names = (:ν, :A, :k, :τ)

function loglike(data, ν, A, k, τ)
    dist = LBA(; ν, A, k, τ)
    return logpdf.(dist, data...) |> sum
end

model = DEModel(;
    sample_prior,
    prior_loglike,
    loglike,
    data = (choice, rt),
    names
)

de = DE(; sample_prior, sample_prior, bounds, burnin = 1500, n_groups = 3, Np = 15)
n_iter = 3000
chain = sample(model, de, MCMCThreads(), n_iter, progress = true)
