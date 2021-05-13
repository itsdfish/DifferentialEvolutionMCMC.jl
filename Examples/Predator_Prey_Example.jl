using DifferentialEquations, StatsBase, StatsPlots
using DifferentialEvolutionMCMC, Distributions, Random, StatsBase, LabelledArrays

Random.seed!(42)

function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, γ, δ  = p
    du[1] = (α - β * y) * x 
    du[2] = (δ * x - γ) * y 
end

# true parameters
p = [1.5, 1.0, 3.0, 1.0]
# initial state values 
u0 = [1.0,1.0]
problem  = ODEProblem(lotka_volterra, u0, (0.0,10.0), p)
sol = solve(problem, Tsit5())
plot(sol)

sol1 = solve(problem, Tsit5(),saveat=0.1)
data = Array(sol1) + 0.5 * randn(size(Array(sol1)))
plot(sol1, alpha = 0.3, legend = false); scatter!(sol1.t, data')

priors = (
    α = (truncated(Normal(1.5, 0.5), 0.5 ,2.5),),
    β = (truncated(Normal(1.2, 0.5), 0, 2),),
    γ = (truncated(Normal(3.0, 0.5), 1, 4),),
    δ = (truncated(Normal(1.0, 0.5), 0, 2),),
    σ = (InverseGamma(2, 3),)
)

bounds = (
    (.5,2.5),
    (0, 2),
    (1,4),
    (0,2),
    (0,Inf)
)

# Log Likelihood function
function loglik(data, problem, θ...)
    prob = remake(problem, p=θ)
    σ = θ[end]
    predicted = solve(prob, Tsit5(), saveat=0.1)
    LL = 0.0
    for i in 1:length(predicted)
        LL += logpdf(MvNormal(predicted[i], σ), data[:,i])
    end
    return LL
end

model = DEModel(problem; priors, model=loglik, data)
de = DE(;bounds, burnin=1000, priors)
n_iter = 3000
@time chains = sample(model, de, n_iter, progress=true)

chain_array = Array(chains)
pl = scatter(sol1.t, data');
for k in 1:300 
    resol = solve(remake(problem, p=chain_array[rand(1:1500), 1:4]),Tsit5(),saveat=0.1)
    plot!(resol, alpha=0.1, color = "#BBBBBB", legend = false)
end

plot!(sol1, w=1, legend = false)