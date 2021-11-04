cd(@__DIR__)
using DifferentialEvolutionMCMC, Random, Distributions

Random.seed!(50514)

n_μ = 30
n_d = 100
μs = rand(Normal(0.0, 1.0), n_μ)
data = rand(MvNormal(μs, 1.0), n_d)

function sample_prior()
    μ = rand(Normal(0, 1), n_μ)
    σ = rand(truncated(Cauchy(0, 1), 0, Inf))
    return as_union([μ,σ])
end

function prior_loglike(μ, σ)
    LL = 0.0
    LL += sum(logpdf.(Normal(0, 1), μ))
    LL += logpdf(truncated(Cauchy(0, 1), 0, Inf), σ)
    return LL
end

bounds = ((-Inf,Inf),(0.0,Inf))


function loglike(data, μs, σ)
    return sum(logpdf(MvNormal(μs, σ), data))
end

names = (:μ,:σ)

model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)


de = DE(;
    bounds, 
    sample = resample,
    burnin = 5000, 
    n_initial=(n_μ+1)*10,
    Np = 3,
    n_groups = 1,
    θsnooker=.1
)
n_iter = 50_000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)