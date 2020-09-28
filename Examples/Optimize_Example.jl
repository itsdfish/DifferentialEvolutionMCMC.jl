cd(@__DIR__)
using Revise, DifferentialEvolutionMCMC, Random, Distributions

Random.seed!(50514)

priors = (
    μ = (Uniform(-10, 10),),
    σ = (Uniform(0, 10),)
)

bounds = ((-Inf,Inf),(0.0,Inf))

data = rand(Normal(0.0, 1.0), 50)

function loglike(μ, σ, data)
    return sum(logpdf.(Normal(μ, σ), data))
end

loglike(θ) = loglike(θ..., data)

model = DEModel(priors=priors, model=loglike)

de = DE(bounds=bounds, Np=5, n_groups=2, update_particle! = greedy_update!,
    evaluate_fitness! = evaluate_fun!)
n_iter = 2000
particles = optimize(model, de, MCMCThreads(), n_iter, progress=true)
results = get_optimal(model, particles)
println(results)
