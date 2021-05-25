cd(@__DIR__)
using DifferentialEvolutionMCMC, Random, Distributions

Random.seed!(50514)

n_μ = 30
n_d = 100
μs = rand(Normal(0.0, 1.0), n_μ)
data = rand(MvNormal(μs, 1.0), n_d)

priors = (
    μ = (Normal(0, 10), n_μ),
    σ = (Truncated(Cauchy(0, 1), 0.0, Inf),)
)
bounds = ((-Inf,Inf),(0.0,Inf))


function loglike(data, μs, σ)
    return sum(logpdf(MvNormal(μs, σ), data))
end

model = DEModel(; priors, model=loglike, data)

de = DE(;bounds, burnin=5000, priors, sample=resample, n_initial=(n_μ+1)*10, 
    Np=3, n_groups=1, θsnooker=.1)
n_iter = 50_000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
