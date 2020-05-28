cd(@__DIR__)
using AdvancedPS, Random, KernelDensity, Distributions
include("KDE.jl")

function loglike(μ, σ, data)
    simdata = rand(Normal(μ, σ), 10_000)
    kd = kernel(simdata)
    dist = InterpKDE(kd)
    #println("proposal: ",μ," ",σ)
    like = max.(1e-10, pdf(dist, data))
    return sum(log.(like))
end

priors = (
    μ=(Normal(0, 10),),
    σ=(Truncated(Cauchy(0, 1), 0.0, Inf),)
)

bounds = ((-Inf,Inf),(0.0,Inf))

data = rand(Normal(0.0, 1.0), 50)

model = DEModel(priors=priors, model=x->loglike(x..., data))

de = DE(bounds=bounds, visualize=false, burnin=1000, priors=priors, progress=true)
n_iter = 2000
chains = sample(model, de, n_iter)
println(chains)
