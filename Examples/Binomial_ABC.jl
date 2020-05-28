cd(@__DIR__)
using AdvancedPS, Random, Parameters, Distributions

priors = (
    θ=(Beta(1, 1),),
)

bounds = ((0,1),)

N = 10
k = rand(Binomial(N, .5))
data = (N=N,k=k)

function loglike(θ, data)
    @unpack N,k = data
    n_sim = 10^4
    counter(_) = rand(Binomial(N, θ)) == k ? (return 1) : (return 0)
    cnt = mapreduce(counter, +, 1:n_sim)
    return log(cnt/n_sim)
end

# loglike(θ, data) = logpdf(Binomial(data.N, θ), data.k)

loglike(θ) = loglike(θ..., data)

model = DEModel(priors=priors, model=loglike)

de = DE(bounds=bounds, burnin=1000, priors=priors, σ=.01)
n_iter = 2000
@elapsed chains = sample(model, de, MCMCThreads(), n_iter, progress=true)
println(chains)
