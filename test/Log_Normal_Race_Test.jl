using DifferentialEvolutionMCMC, Test, Random, Turing, Parameters, Distributions
using SequentialSamplingModels, LinearAlgebra
import DifferentialEvolutionMCMC: select_groups, select_particles, shift_particles!, sample_init
Random.seed!(59391)

dist = LNR(μ=[-2.,-2.,-3.,-3], σ=1.0, ϕ=.5)
data = rand(dist, 100)

function loglike(data, μ, σ, ϕ)
    dist = LNR(μ, σ, ϕ)
    return sum(logpdf.(dist, data))
end

function prior_loglike(μ, σ, ϕ)
    LL = 0.0
    LL += logpdf(Normal(0, 3), 4), μ)
    LL += logpdf(truncated(Cauchy(0, 1), 0.0, Inf), σ)
    LL += logpdf(Uniform(0., min_rt), ϕ)
end

function sample_loglike()
    LL = 0.0
    μ = rand(Normal(0, 3), 4)
    σ = logpdf(truncated(Cauchy(0, 1), 0.0, Inf))
    ϕ = logpdf(Uniform(0, min_rt))
    return as_union([μ,σ,ϕ])
end

min_rt = minimum(x -> x[2], data)
bounds = ((-Inf,0.),(1e-10,Inf),(0.,min_rt))
# model = DEModel(;priors, model=loglike, data)
# de = DE(;priors, bounds, burnin=2000)
# n_iter = 4000
# chains = sample(model, de, n_iter)

names = (:μ,:σ,:ϕ)

model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)


de = DE(;bounds, burnin=2000, Np=24)
n_iter = 4000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)


μ_de = describe(chains)[1][:,:mean]
σ_de = describe(chains)[1][:,:std]
rhat = describe(chains)[1][:,:rhat]

@model model1(data) = begin
    min_rt = minimum(x -> x[2], data)
    μ ~ MvNormal(zeros(4), I * 3^2)
    σ ~ truncated(Cauchy(0, 1), 0.0, Inf)
    ϕ ~ Uniform(0.0, min_rt)
    data ~ LNR(μ=μ, σ=σ, ϕ=ϕ)
end

chn = sample(model1(data), NUTS(1000, .85), 2000)
μ_nuts = describe(chn)[1][:,:mean]
σ_nuts = describe(chn)[1][:,:std]

@test all(isapprox.(rhat, fill(1.0, 6), atol=.05))
@test all(isapprox.(μ_nuts, μ_de, rtol=.05))
@test all(isapprox.(σ_nuts, σ_de, rtol=.05))