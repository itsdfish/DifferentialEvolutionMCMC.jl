cd(@__DIR__)
using DifferentialEvolutionMCMC, Random, Distributions
Random.seed!(88484)

prior_loglike(θ) = logpdf(Beta(1, 1), θ)

sample_prior() = rand(Beta(1, 1))

bounds = ((0,1),)
names = (:θ,)

N = 10
k = rand(Binomial(N, .5))
data = (N = N,k = k)

function loglike(data, θ)
    (;N,k) = data
    n_sim = 10^4
    counter(_) = rand(Binomial(N, θ)) == k ? 1 : 0
    cnt = mapreduce(counter, +, 1:n_sim)
    return log(cnt / n_sim)
end

# loglike(θ, data) = logpdf(Binomial(data.N, θ), data.k)

model = DEModel(; 
    sample_prior, 
    prior_loglike, 
    loglike, 
    data,
    names
)

de = DE(;sample_prior, bounds, burnin=1000, Np=3, σ=.01)
n_iter = 2000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)