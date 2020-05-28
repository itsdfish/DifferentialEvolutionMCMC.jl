cd(@__DIR__)
using AdvancedPS, Random, Distributions

Random.seed!(50514)

priors = (
    μ=(Normal(0, 10),),
    σ=(Truncated(Cauchy(0, 1), 0.0, Inf),)
)

bounds = ((-Inf,Inf),(0.0,Inf))

data = rand(Normal(0.0, 1.0), 50)

function loglike(μ, σ, data)
    return sum(logpdf.(Normal(μ, σ), data))
end

loglike(θ) = loglike(θ..., data)

model = DEModel(priors=priors, model=loglike)

de = DE(bounds=bounds, burnin=1000, priors=priors, progress=true)
n_iter = 2000
chains = psample(model, de, n_iter)
println(chains)
