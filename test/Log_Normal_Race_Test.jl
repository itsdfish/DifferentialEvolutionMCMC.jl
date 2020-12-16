using DifferentialEvolutionMCMC, Test, Random, Turing, Parameters, Distributions
using SequentialSamplingModels
import DifferentialEvolutionMCMC: select_groups, select_particles, shift_particles!, sample_init
Random.seed!(59391)

dist = LNR(μ=[-2.,-2.,-3.,-3], σ=1.0, ϕ=.5)
data = rand(dist, 100)

function loglike(μ, σ, ϕ, data)
    dist = LNR(μ=μ, σ=σ, ϕ=ϕ)
    return sum(logpdf.(dist, data))
end

minRT = minimum(x -> x[2], data)
priors = (μ = (Normal(0, 3), 4),σ = (truncated(Cauchy(0, 1), 0.0, Inf),),
    ϕ = (Uniform(0., minRT),))
bounds = ((-Inf,0.),(1e-10,Inf),(0.,minRT))
model = DEModel(priors=priors, model=loglike, data=data)
de = DE(;priors=priors, bounds=bounds, burnin=2000)
n_iter = 4000
chains = sample(model, de, n_iter)
μ_de = describe(chains)[1][:,:mean]
σ_de = describe(chains)[1][:,:std]
rhat = describe(chains)[1][:,:rhat]

@model model1(data) = begin
    minRT = minimum(x -> x[2], data)
    μ ~ MvNormal(zeros(4), 3)
    σ ~ truncated(Cauchy(0, 1), 0.0, Inf)
    ϕ ~ Uniform(0.0, minRT)
    data ~ LNR(μ=μ, σ=σ, ϕ=ϕ)
end

chn = sample(model1(data), NUTS(1000, .85), 2000)
μ_nuts = describe(chn)[1][:,:mean]
σ_nuts = describe(chn)[1][:,:std]

@test all(isapprox.(rhat, fill(1.0, 6), atol=.05))
@test all(isapprox.(μ_nuts, μ_de, rtol=.05))
@test all(isapprox.(σ_nuts, σ_de, rtol=.05))