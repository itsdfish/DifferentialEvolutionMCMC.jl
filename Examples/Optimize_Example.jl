cd(@__DIR__)
using Revise, DifferentialEvolutionMCMC, Random, Distributions
import DifferentialEvolutionMCMC: minimize!

Random.seed!(50514)

priors = (
    x = (Uniform(-5, 5), 2),
)

bounds = ((-5.0,5.0),)

function rosenbrock2d(data, x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

model = DEModel(; priors, model=rosenbrock2d, data=nothing)

de = DE(bounds=bounds, Np=6, n_groups=1, update_particle! = minimize!,
    evaluate_fitness! = evaluate_fun!)
n_iter = 10000
particles = optimize(model, de, MCMCThreads(), n_iter, progress=true);
results = get_optimal(de, model, particles)
println(results)