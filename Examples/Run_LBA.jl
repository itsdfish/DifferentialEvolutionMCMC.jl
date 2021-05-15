cd(@__DIR__)
using DifferentialEvolutionMCMC, Random, Parameters
using SequentialSamplingModels, Distributions
Random.seed!(88484)

dist = LBA(ν=[3.0,2.0], A = .8, k = .2, τ = .3) 
choice,rt = rand(dist, 100)
min_rt = minimum(rt)

priors = (
    ν = (Normal(1, 5), 2),
    A = (Normal(.8, .2),),
    k = (Normal(.2, .1),),
    τ = (Uniform(0, min_rt),)
)

bounds = ((0.0,Inf), (0.0,Inf), (0.0,Inf), (0.0,min_rt))

function loglike(data, ν, A, k, τ)
    dist = LBA(;ν, A, k, τ) 
    return logpdf.(dist, data...) |> sum
end

model = DEModel(;priors, model=loglike, data=(choice,rt))

de = DE(;bounds, burnin=1500, priors)
n_iter = 3000
@elapsed chain = sample(model, de, MCMCThreads(), n_iter, progress=true)
