# Note: poor performance



# this may not be working properly


cd(@__DIR__)
using AdvancedPS, Random, StatsBase, Distributions

N = 30
# Parameters for each cluster, we assume that each cluster is Gaussian distributed in the example.
μs = [-3.5, 0.0]
# Construct the data points.
data = mapreduce(c -> rand(MvNormal([μs[c], μs[c]], 1.), N), hcat, 1:2)

priors = (
    idx=(Categorical([.5,.5]),N),
    μs = (Normal(0,1),2),
)

bounds = ((1,2),(0.0,Inf))

function loglike(idx, μs, data)
    LL = 0.0
    for (i,id) in enumerate(idx)
        LL += logpdf(MvNormal([μs[id], μs[id]], 1.0), data[:,i])
    end
    return LL
end

loglike(θ) = loglike(θ..., data)

model = DEModel(priors=priors, model=loglike)

de = DE(bounds=bounds, burnin=1000, priors=priors, progress=true, Np=15)
n_iter = 2000
@elapsed chains = sample(model, de, n_iter)
println(chains)
