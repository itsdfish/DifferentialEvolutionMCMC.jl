cd(@__DIR__)
using Revise, DifferentialEvolutionMCMC, Random, Distributions
import DifferentialEvolutionMCMC: minimize!

Random.seed!(50514)

function sample_prior()
    return [rand(Uniform(-5, 5), 2)]
 end

 function rastrigin(data, x)
     A = 10.0
     n = length(x)
     y = A * n
     for  i in 1:n
         y +=  + x[i]^2 - A * cos(2 * π * x[i])
     end
     return y 
 end

bounds = ((-5.0,5.0),)
names = (:x,)

model = DEModel(; 
    sample_prior, 
    loglike = rastrigin, 
    data = nothing,
    names
)

de = DE(;
    sample_prior,
    bounds,
    Np = 6, 
    n_groups = 1, 
    update_particle! = minimize!,
    evaluate_fitness! = evaluate_fun!
)



n_iter = 10_000
particles = optimize(model, de, n_iter, progress=true);
results = get_optimal(de, model, particles)
println(results)