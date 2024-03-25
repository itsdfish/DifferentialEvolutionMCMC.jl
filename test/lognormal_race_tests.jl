using DifferentialEvolutionMCMC, Test, Random, Parameters, Distributions
using SequentialSamplingModels, LinearAlgebra
using Turing
Random.seed!(9918)

dist = LNR(ν = [-2.0, -2.0, -3.0, -3], σ = fill(1.0, 4), τ = 0.5)
data = rand(dist, 100)

function loglike(data, ν, τ)
    dist = LNR(; ν, τ)
    return sum(logpdf(dist, data))
end

function prior_loglike(ν, τ)
    LL = 0.0
    LL += sum(logpdf.(Normal(0, 3), ν))
    LL += logpdf(Uniform(0.0, min_rt), τ)
end

function sample_prior()
    LL = 0.0
    ν = rand(Normal(0, 3), 4)
    τ = rand(Uniform(0, min_rt))
    return as_union([ν, τ])
end

min_rt = minimum(x -> x[2], data)

bounds = ((-Inf, Inf), (0.0, min_rt))
names = (:ν, :τ)

model = DEModel(;
    sample_prior,
    prior_loglike,
    loglike,
    data,
    names
)

de = DE(; sample_prior, bounds, burnin = 2000, Np = 24, n_groups = 4)
n_iter = 5000
chains = sample(model, de, MCMCThreads(), n_iter, progress = true)

μ_de = describe(chains)[1][:, :mean]
σ_de = describe(chains)[1][:, :std]
rhat = describe(chains)[1][:, :rhat]

import Distributions: loglikelihood

loglikelihood(d::LNR, data::Vector{<:Tuple}) = sum(logpdf.(d, data))

@model model1(data) = begin
    min_rt = minimum(x -> x[2], data)
    ν ~ MvNormal(zeros(4), I * 3^2)
    τ ~ Uniform(0.0, min_rt)
    data ~ LNR(; ν, τ)
end

Random.seed!(68541)
chn = sample(model1(data), NUTS(1500, 0.85), 4000)
μ_nuts = describe(chn)[1][:, :mean]
σ_nuts = describe(chn)[1][:, :std]

@test all(isapprox.(rhat, fill(1.0, 5), atol = 0.05))
@test all(isapprox.(μ_nuts, μ_de, rtol = 0.05))
@test all(isapprox.(σ_nuts, σ_de, rtol = 0.05))
