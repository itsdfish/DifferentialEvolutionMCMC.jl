using DifferentialEvolutionMCMC, Test, Random, Turing, Parameters, Distributions
import DifferentialEvolutionMCMC: select_groups, select_particles, shift_particles!, sample_init
Random.seed!(973536)
priors = (
    μ = (Normal(0, 10),),
    σ = (truncated(Cauchy(0, 1), 0.0, Inf),)
)

bounds = ((-Inf,Inf),(0.0,Inf))

data = rand(Normal(0.0, 1.0), 100)

function loglike(data, μ, σ)
    return sum(logpdf.(Normal(μ, σ), data))
end

model = DEModel(priors=priors, model=loglike, data=data)
de = DE(;priors=priors, bounds=bounds, burnin=1500)
n_iter = 3000
chains = sample(model, de, n_iter)
μ_de = describe(chains)[1][:,:mean]
σ_de = describe(chains)[1][:,:std]
rhat = describe(chains)[1][:,:rhat]

@model model1(data) = begin
    μ ~ Normal(0, 10)
    σ ~ truncated(Cauchy(0, 1), 0.0, Inf)
    for i in 1:length(data)
        data[i] ~ Normal(μ, σ)
    end
end
chn = sample(model1(data), NUTS(1000, .85), 2000)
μ_nuts = describe(chn)[1][:,:mean]
σ_nuts = describe(chn)[1][:,:std]

@test all(isapprox.(rhat, fill(1.0, 2), atol=.05))
@test all(isapprox.(μ_nuts, μ_de, atol=.01))
@test all(isapprox.(σ_nuts, σ_de, atol=.01))