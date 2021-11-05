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

function prior_loglike(α, β, γ, δ, σ)
    LL = 0.0
    LL += logpdf(truncated(Normal(1.5, 0.5), 0.5 ,2.5), α)
    LL += logpdf(truncated(Normal(1.2, 0.5), 0, 2), β)
    LL += logpdf(truncated(Normal(3.0, 0.5), 1, 4), γ)
    LL += logpdf(truncated(Normal(1.0, 0.5), 0, 2), δ)
    LL += logpdf(InverseGamma(2, 3), σ)
    return LL
end

function sample_prior()
    α = rand(truncated(Normal(1.5, 0.5), 0.5 ,2.5))
    β = rand(truncated(Normal(1.2, 0.5), 0, 2))
    γ = rand(truncated(Normal(3.0, 0.5), 1, 4))
    δ = rand(truncated(Normal(1.0, 0.5), 0, 2))
    σ = rand(InverseGamma(2, 3))
    return [α, β, γ, δ, σ]
end

bounds = (
    (.5,2.5),
    (0, 2),
    (1,4),
    (0,2),
    (0,Inf)
)

names = (:α,:β,:γ,:δ,:σ)


# Log Likelihood function
function loglike(data, problem, θ...)
    prob = remake(problem, p=θ)
    σ = θ[end]
    predicted = solve(prob, Tsit5(), saveat=0.1)
    LL = 0.0
    for i in 1:length(predicted)
        LL += logpdf(MvNormal(predicted[i], σ), data[:,i])
    end
    return LL
end

model = DEModel(
    problem; 
    loglike,
    prior_loglike,
    sample_prior,
    data,
    names
)

de = DE(;bounds, burnin=1000, Np=12, n_groups=3)
n_iter = 3000
chains = sample(model, de, MCMCThreads(), n_iter, progress=true)

chain_array = Array(chains)
pl = plot(grid=false)
for k in 1:300 
    p = chain_array[rand(1:2000),:]
    resol = solve(remake(problem, p=p[1:4]),Tsit5(), saveat=0.1)
    sim_data = Array(resol) + p[end] * randn(size(Array(resol)))
    plot!(resol.t, sim_data', alpha=0.1, color = "#BBBBBB", legend = false)
end
scatter!(sol1.t, data');
plot!(sol1, w=1, legend = false)