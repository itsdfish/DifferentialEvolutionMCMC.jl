using DifferentialEvolutionMCMC, Test, Random, Turing, Distributions
Random.seed!(973536)

bounds = ((-Inf,Inf),(0.0,Inf))

data = rand(Normal(0.0, 1.0), 100)

function prior_loglike(μ, σ)
    LL = 0.0
    LL += logpdf(Normal(0, 10), μ)
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
    return LL
end

function sample_prior()
    μ = rand(Normal(0, 10))
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

de = DE(;sample_prior, bounds, burnin=1500, Np=6)
n_iter = 3000
chains = sample(model, de, n_iter, progress=true)
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